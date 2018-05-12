"""
This version of the Behler-Parinello is aperiodic,non-sparse.
It's being developed to explore alternatives to symmetry functions. It's not for production use.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import sys
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle

from ..TFDescriptors.RawSH import *
from ..TFDescriptors.RawSymFunc import *
from ..ElementData import *
from ..Math.TFMath import *
from tensorflow.python.client import timeline

class UniversalNetwork(object):
	"""
	Base class for Behler-Parinello network using embedding from RawEmbeddings.py
	Do not use directly, only for inheritance to derived classes
	also has sparse evaluation using an updated version of the
	neighbor list, and a polynomial cutoff coulomb interaction.
	"""
	def __init__(self, mol_set_name=None, name=None):
		"""
		Args:
			mol_set (TensorMol.MSet object): a class which holds the training data
			name (str): a name used to recall this network

		Notes:
			if name != None, attempts to load a previously saved network, otherwise assumes a new network
		"""
		self.tf_precision = eval(PARAMS["tf_prec"])
		self.hidden_layers = PARAMS["HiddenLayers"]
		self.learning_rate = PARAMS["learning_rate"]
		self.weight_decay = PARAMS["weight_decay"]
		self.momentum = PARAMS["momentum"]
		self.max_steps = PARAMS["max_steps"]
		self.batch_size = PARAMS["batch_size"]
		self.max_checkpoints = PARAMS["max_checkpoints"]
		self.path = PARAMS["networks_directory"]
		self.train_gradients = PARAMS["train_gradients"]
		self.train_charges = PARAMS["train_charges"]
		self.profiling = PARAMS["Profiling"]
		self.activation_function_type = PARAMS["NeuronType"]
		self.test_ratio = PARAMS["TestRatio"]
		self.element_codes = ELEMENTCODES
		self.element_codepairs = np.zeros((self.element_codes.shape[0]*(self.element_codes.shape[0]+1)/2, self.element_codes.shape[1]))
		self.codepair_idx = np.zeros((self.element_codes.shape[0], self.element_codes.shape[0]), dtype=np.int32)
		counter = 0
		for i in range(len(self.element_codes)):
			for j in range(i, len(self.element_codes)):
				self.codepair_idx[i,j] = counter
				self.codepair_idx[j,i] = counter
				self.element_codepairs[counter] = self.element_codes[i] * self.element_codes[j]
				counter += 1
		self.assign_activation()

		#Reloads a previous network if name variable is not None
		if name != None:
			self.name = name
			self.load_network()
			LOGGER.info("Reloaded network from %s", self.network_directory)
			return

		#Data parameters
		self.mol_set_name = mol_set_name
		self.mol_set = MSet(self.mol_set_name)
		self.mol_set.Load()
		self.elements = self.mol_set.AtomTypes()
		self.max_num_atoms = self.mol_set.MaxNAtom()
		self.num_molecules = len(self.mol_set.mols)
		self.energy_fit = np.zeros((self.mol_set.max_atomic_num()+1))
		self.charge_fit = np.zeros((self.mol_set.max_atomic_num()+1))
		energy_fit, charge_fit = self.mol_set.RemoveElementAverages()
		for element in energy_fit.keys():
			self.energy_fit[element] = energy_fit[element]
			self.charge_fit[element] = charge_fit[element]
		self.step = 0
		self.test_freq = PARAMS["test_freq"]
		self.network_type = "SF_Universal"
		self.name = self.network_type+"_"+self.mol_set_name+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.network_directory = PARAMS["networks_directory"]+self.name
		self.set_symmetry_function_params()
		LOGGER.info("learning rate: %f", self.learning_rate)
		LOGGER.info("batch size:    %d", self.batch_size)
		LOGGER.info("max steps:     %d", self.max_steps)
		return

	def __getstate__(self):
		state = self.__dict__.copy()
		remove_vars = ["mol_set", "activation_function", "xyz_data", "Z_data", "energy_data", "charges_data",
						"num_atoms_data", "gradient_data"]
		for var in remove_vars:
			try:
				del state[var]
			except:
				pass
		return state

	def set_symmetry_function_params(self):
		self.element_pairs = np.array([[self.elements[i], self.elements[j]] for i in range(len(self.elements)) for j in range(i, len(self.elements))])
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]

		#Define radial grid parameters
		num_radial_rs = PARAMS["AN1_num_r_Rs"]
		self.radial_cutoff = PARAMS["AN1_r_Rc"]
		self.radial_rs = self.radial_cutoff * np.linspace(0, (num_radial_rs - 1.0) / num_radial_rs, num_radial_rs)

		#Define angular grid parameters
		num_angular_rs = PARAMS["AN1_num_a_Rs"]
		num_angular_theta_s = PARAMS["AN1_num_a_As"]
		self.angular_cutoff = PARAMS["AN1_a_Rc"]
		self.theta_s = np.pi * np.linspace(0, (num_angular_theta_s - 1.0) / num_angular_theta_s, num_angular_theta_s)
		self.angular_rs = self.angular_cutoff * np.linspace(0, (num_angular_rs - 1.0) / num_angular_rs, num_angular_rs)
		return

	def assign_activation(self):
		LOGGER.debug("Assigning Activation Function: %s", PARAMS["NeuronType"])
		try:
			if self.activation_function_type == "relu":
				self.activation_function = tf.nn.relu
			elif self.activation_function_type == "elu":
				self.activation_function = tf.nn.elu
			elif self.activation_function_type == "selu":
				self.activation_function = self.selu
			elif self.activation_function_type == "softplus":
				self.activation_function = tf.nn.softplus
			elif self.activation_function_type == "shifted_softplus":
				self.activation_function = self.shifted_softplus
			elif self.activation_function_type == "tanh":
				self.activation_function = tf.tanh
			elif self.activation_function_type == "sigmoid":
				self.activation_function = tf.sigmoid
			elif self.activation_function_type == "sigmoid_with_param":
				self.activation_function = self.sigmoid_with_param
			else:
				print ("unknown activation function, set to relu")
				self.activation_function = tf.nn.relu
		except Exception as Ex:
			print(Ex)
			print ("activation function not assigned, set to relu")
			self.activation_function = tf.nn.relu
		return

	def shifted_softplus(self, x):
		return tf.nn.softplus(x) - tf.cast(tf.log(2.0), self.tf_precision)

	def start_training(self):
		self.load_data_to_scratch()
		self.compute_normalization()
		self.save_network()
		self.train_prepare()
		self.train()

	def restart_training(self):
		self.reload_set()
		self.load_data()
		self.train_prepare(restart=True)
		self.train()

	def train(self):
		for i in range(self.max_steps):
			self.step += 1
			self.train_step(self.step)
			if self.step%self.test_freq==0:
				test_loss = self.test_step(self.step)
				if self.step == self.test_freq:
					self.best_loss = test_loss
					self.save_checkpoint(self.step)
				elif test_loss < self.best_loss:
					self.best_loss = test_loss
					self.save_checkpoint(self.step)
		self.sess.close()
		return

	def save_checkpoint(self, step):
		checkpoint_file = os.path.join(self.network_directory,self.name+'-checkpoint')
		LOGGER.info("Saving checkpoint file %s", checkpoint_file)
		self.saver.save(self.sess, checkpoint_file, global_step=step)
		return

	def save_network(self):
		print("Saving TFInstance")
		f = open(PARAMS["networks_directory"]+self.name+".tfn","wb")
		if sys.version_info[0] < 3:
			pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			pickle.dump(self, f)
		f.close()
		return

	def load_network(self):
		LOGGER.info("Loading TFInstance")
		network = pickle.load(open(self.path+"/"+self.name+".tfn","rb"))
		self.__dict__.update(network.__dict__)
		return

	def reload_set(self):
		"""
		Recalls the MSet to build training data etc.
		"""
		self.mol_set = MSet(self.mol_set_name)
		self.mol_set.Load()
		return

	def load_data(self):
		if (self.mol_set == None):
			try:
				self.reload_set()
			except Exception as Ex:
				print("TensorData object has no molecule set.", Ex)
		self.xyz_data = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype = np.float64)
		self.Z_data = np.zeros((self.num_molecules, self.max_num_atoms), dtype = np.int32)
		self.charges_data = np.zeros((self.num_molecules, self.max_num_atoms), dtype = np.float64)
		self.num_atoms_data = np.zeros((self.num_molecules), dtype = np.int32)
		self.energy_data = np.zeros((self.num_molecules), dtype = np.float64)
		self.gradient_data = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(self.mol_set.mols):
			self.xyz_data[i][:mol.NAtoms()] = mol.coords
			self.Z_data[i][:mol.NAtoms()] = mol.atoms
			self.charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			self.energy_data[i] = mol.properties["energy"]
			self.gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			self.num_atoms_data[i] = mol.NAtoms()
		return

	def load_data_to_scratch(self):
		"""
		Reads built training data off disk into scratch space.
		Divides training and test data.
		Normalizes inputs and outputs.
		note that modifies my MolDigester to incorporate the normalization
		Initializes pointers used to provide training batches.

		Args:
			random: Not yet implemented randomization of the read data.

		Note:
			Also determines mean stoichiometry
		"""
		self.load_data()
		self.num_test_cases = int(self.test_ratio * self.num_molecules)
		self.num_train_cases = int(self.num_molecules - self.num_test_cases)
		case_idxs = np.arange(int(self.num_molecules))
		np.random.shuffle(case_idxs)
		self.train_idxs = case_idxs[:int(self.num_molecules - self.num_test_cases)]
		self.test_idxs = case_idxs[int(self.num_molecules - self.num_test_cases):]
		self.train_pointer, self.test_pointer = 0, 0
		if self.batch_size > self.num_train_cases:
			raise Exception("Insufficent training data to fill a training batch.\n"\
					+str(self.num_train_cases)+" cases in dataset with a batch size of "+str(self.batch_size))
		if self.batch_size > self.num_test_cases:
			raise Exception("Insufficent testing data to fill a test batch.\n"\
					+str(self.num_test_cases)+" cases in dataset with a batch size of "+str(self.batch_size))
		LOGGER.debug("Number of training cases: %i", self.num_train_cases)
		LOGGER.debug("Number of test cases: %i", self.num_test_cases)
		return

	def get_train_batch(self, batch_size):
		if self.train_pointer + batch_size >= self.num_train_cases:
			np.random.shuffle(self.train_idxs)
			self.train_pointer = 0
		self.train_pointer += batch_size
		batch_xyzs = self.xyz_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]]
		batch_Zs = self.Z_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]]
		nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 19.0, self.max_num_atoms, False, False)
		batch_data = []
		batch_data.append(batch_xyzs)
		batch_data.append(batch_Zs)
		batch_data.append(nn_pairs)
		batch_data.append(nn_triples)
		batch_data.append(coulomb_pairs)
		batch_data.append(self.num_atoms_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.energy_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.gradient_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.charges_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		return batch_data

	def get_test_batch(self, batch_size):
		if self.test_pointer + batch_size >= self.num_test_cases:
			self.test_pointer = 0
		self.test_pointer += batch_size
		batch_xyzs = self.xyz_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]]
		batch_Zs = self.Z_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]]
		nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 19.0, self.max_num_atoms, False, False)
		batch_data = []
		batch_data.append(batch_xyzs)
		batch_data.append(batch_Zs)
		batch_data.append(nn_pairs)
		batch_data.append(nn_triples)
		batch_data.append(coulomb_pairs)
		batch_data.append(self.num_atoms_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.energy_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.gradient_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.charges_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		return batch_data

	def variable_with_weight_decay(self, shape, stddev, weight_decay, name = None):
		"""
		Creates random tensorflow variable from a truncated normal distribution with weight decay

		Args:
			name: name of the variable
			shape: list of ints
			stddev: standard deviation of a truncated Gaussian
			wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.

		Returns:
			Variable Tensor

		Notes:
			Note that the Variable is initialized with a truncated normal distribution.
			A weight decay is added only if one is specified.
		"""
		variable = tf.Variable(tf.truncated_normal(shape, stddev = stddev, dtype = self.tf_precision), name = name)
		if weight_decay is not None:
			weightdecay = tf.multiply(tf.nn.l2_loss(variable), weight_decay, name='weight_loss')
			tf.add_to_collection('energy_losses', weightdecay)
		return variable

	def fill_feed_dict(self, batch_data):
		"""
		Fill the tensorflow feed dictionary.

		Args:
			batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
			and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

		Returns:
			Filled feed dictionary.
		"""
		pl_list = [self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl, self.coulomb_pairs_pl,
					self.num_atoms_pl, self.energy_pl, self.gradients_pl, self.charges_pl]
		feed_dict={i: d for i, d in zip(pl_list, batch_data)}
		return feed_dict

	def energy_inference(self, embed, atom_codes, indices):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		variables=[]
		with tf.variable_scope("energy_network", reuse=tf.AUTO_REUSE):
			code_kernel1 = tf.get_variable(name="CodeKernel1", shape=(4, 4), dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(4, 4), dtype=self.tf_precision)
			variables.append(code_kernel1)
			variables.append(code_kernel2)
			coded_weights = tf.matmul(atom_codes, code_kernel1)
			coded_embed = tf.einsum('ijk,ij->ijk', embed, coded_weights)
			coded_embed = tf.reshape(tf.einsum('ijk,jl->ilk', coded_embed, code_kernel2), [tf.shape(embed)[0], -1])
			for i in range(len(self.hidden_layers)):
				if i == 0:
					with tf.name_scope('hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(coded_embed, weights) + biases)
						variables.append(weights)
						variables.append(biases)
				else:
					with tf.name_scope('hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(activations, weights) + biases)
						variables.append(weights)
						variables.append(biases)
			with tf.name_scope('regression_linear'):
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
						stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([2], dtype=self.tf_precision), name='biases')
				outputs = tf.matmul(activations, weights) + biases
				variables.append(weights)
				variables.append(biases)
				atom_nn_energy = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
		return atom_nn_energy, atom_nn_charges, variables

	def charge_inference(self, embed, atom_codes, Zs, n_atoms):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		indices = tf.where(tf.not_equal(Zs, 0))
		variables=[]
		with tf.variable_scope("charge_network", reuse=tf.AUTO_REUSE):
			code_kernel1 = tf.get_variable(name="CodeKernel", shape=(4, 4),dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(4, 4),dtype=self.tf_precision)
			variables.append(code_kernel1)
			variables.append(code_kernel2)
			coded_weights = tf.matmul(atom_codes, code_kernel1)
			coded_embed = tf.einsum('ijk,ij->ijk', embed, coded_weights)
			coded_embed = tf.reshape(tf.einsum('ijk,jl->ilk', coded_embed, code_kernel2), [tf.shape(embed)[0], -1])
			embed = tf.reshape(embed, [tf.shape(embed)[0], -1])
			for i in range(len(self.hidden_layers)):
				if i == 0:
					with tf.name_scope('hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(embed, weights) + biases)
						variables.append(weights)
						variables.append(biases)
				else:
					with tf.name_scope('hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(activations, weights) + biases)
						variables.append(weights)
						variables.append(biases)
			with tf.name_scope('regression_linear'):
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
						stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
				outputs = tf.squeeze(tf.matmul(activations, weights) + biases, axis=1)
				variables.append(weights)
				variables.append(biases)
				output = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
				excess_charge = tf.reduce_sum(output, axis=1)
				output -= tf.expand_dims(excess_charge / tf.cast(n_atoms, eval(PARAMS["tf_prec"])), axis=-1)
				mask = tf.where(tf.equal(Zs, 0), tf.zeros_like(Zs, dtype=eval(PARAMS["tf_prec"])),
						tf.ones_like(Zs, dtype=eval(PARAMS["tf_prec"])))
				atom_nn_charges = output * mask
		return atom_nn_charges, variables

	def gather_coulomb(self, xyzs, Zs, atom_charges, pairs):
		padding_mask = tf.where(tf.logical_and(tf.not_equal(Zs, 0), tf.reduce_any(tf.not_equal(pairs, -1), axis=-1)))
		central_atom_coords = tf.gather_nd(xyzs, padding_mask)
		central_atom_charge = tf.gather_nd(atom_charges, padding_mask)
		pairs = tf.gather_nd(pairs, padding_mask)
		padded_pairs = tf.equal(pairs, -1)
		tmp_pairs = tf.where(padded_pairs, tf.zeros_like(pairs), pairs)
		gather_pairs = tf.stack([tf.cast(tf.tile(padding_mask[:,:1], [1, tf.shape(pairs)[1]]), tf.int32), tmp_pairs], axis=-1)
		pair_coords = tf.gather_nd(xyzs, gather_pairs)
		dxyzs = tf.expand_dims(central_atom_coords, axis=1) - pair_coords
		pair_mask = tf.where(padded_pairs, tf.zeros_like(pairs), tf.ones_like(pairs))
		dxyzs *= tf.cast(tf.expand_dims(pair_mask, axis=-1), eval(PARAMS["tf_prec"]))
		pair_charges = tf.gather_nd(atom_charges, gather_pairs)
		pair_charges *= tf.cast(pair_mask, eval(PARAMS["tf_prec"]))
		q1q2 = tf.expand_dims(central_atom_charge, axis=-1) * pair_charges
		return dxyzs, q1q2, padding_mask

	def calculate_coulomb_energy(self, dxyzs, q1q2, scatter_idx):
		"""
		Polynomial cutoff 1/r (in BOHR) obeying:
		kern = 1/r at SROuter and LRInner
		d(kern) = d(1/r) (true force) at SROuter,LRInner
		d**2(kern) = d**2(1/r) at SROuter and LRInner.
		d(kern) = 0 (no force) at/beyond SRInner and LROuter

		The hard cutoff is LROuter
		"""
		srange_inner = 4.5
		srange_outer = 7.5
		lrange_inner = 16.
		lrange_outer = 19.
		a, b, c, d, e, f, g, h = -7.25102, 5.35606, -1.45437, 0.213988, -0.0184294, 0.000930203, -0.0000255246, 2.94322e-7
		dist = tf.norm(dxyzs+1.e-16, axis=-1)
		dist *= 1.889725989
		dist2 = dist * dist
		dist3 = dist2 * dist
		dist4 = dist3 * dist
		dist5 = dist4 * dist
		dist6 = dist5 * dist
		dist7 = dist6 * dist
		mrange_kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
		kern = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) / srange_inner, mrange_kern)
		kern = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) / lrange_outer, kern)
		mrange_energy = tf.reduce_sum(kern * q1q2, axis=1)
		lrange_energy = tf.reduce_sum(q1q2, axis=1) / lrange_outer
		coulomb_energy = mrange_energy - lrange_energy
		return tf.reduce_sum(tf.scatter_nd(scatter_idx, coulomb_energy, [self.batch_size, self.max_num_atoms]), axis=-1)

	def optimizer(self, loss, learning_rate, momentum, variables=None):
		"""
		Sets up the training Ops.
		Creates a summarizer to track the loss over time in TensorBoard.
		Creates an optimizer and applies the gradients to all trainable variables.
		The Op returned by this function is what must be passed to the
		`sess.run()` call to cause the model to train.

		Args:
			loss: Loss tensor, from loss().
			learning_rate: the learning rate to use for gradient descent.

		Returns:
			train_op: the tensorflow operation to call for training.
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		if variables == None:
			train_op = optimizer.minimize(loss, global_step=global_step)
		else:
			train_op = optimizer.minimize(loss, global_step=global_step, var_list=variables)
		return train_op

	def loss_op(self, error):
		loss = tf.nn.l2_loss(error)
		return loss

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.num_train_cases
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		train_gradient_loss = 0.0
		train_charge_loss = 0.0
		num_batches = 0
		for ministep in range (0, int(0.025 * Ncase_train/self.batch_size)):
			batch_data = self.get_train_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_gradients and self.train_charges:
				_, summaries, total_loss, energy_loss, gradient_loss, charge_loss = self.sess.run([self.train_op,
				self.summary_op, self.total_loss, self.energy_loss, self.gradient_loss, self.charge_loss], feed_dict=feed_dict)
				train_gradient_loss += gradient_loss
				train_charge_loss += charge_loss
			elif self.train_charges:
				_, summaries, total_loss, energy_loss, charge_loss = self.sess.run([self.train_op,
				self.summary_op, self.total_loss, self.energy_loss, self.charge_loss], feed_dict=feed_dict)
				train_charge_loss += charge_loss
			elif self.train_gradients:
				_, summaries, total_loss, energy_loss, gradient_loss = self.sess.run([self.train_op,
				self.summary_op, self.total_loss, self.energy_loss, self.gradient_loss], feed_dict=feed_dict)
				train_gradient_loss += gradient_loss
			else:
				_, summaries, total_loss, energy_loss = self.sess.run([self.train_op,
				self.summary_op, self.total_loss, self.energy_loss], feed_dict=feed_dict)
			train_loss += total_loss
			train_energy_loss += energy_loss
			num_batches += 1
			self.summary_writer.add_summary(summaries, step * int(Ncase_train/self.batch_size) + ministep)
		train_loss /= num_batches
		train_energy_loss /= num_batches
		train_gradient_loss /= num_batches
		train_charge_loss /= num_batches
		duration = time.time() - start_time
		self.print_epoch(step, duration, train_loss, train_energy_loss, train_gradient_loss, train_charge_loss)
		return

	def test_step(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		test_loss =  0.0
		start_time = time.time()
		Ncase_test = self.num_test_cases
		num_batches = 0
		test_energy_loss = 0.0
		test_gradient_loss = 0.0
		test_charge_loss = 0.0
		test_energy_labels, test_energy_outputs = [], []
		test_force_labels, test_force_outputs = [], []
		test_charge_labels, test_charge_outputs = [], []
		num_atoms_epoch = 0.0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.get_test_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_charges:
				total_energies, energy_labels, gradients, gradient_labels, charges, charge_labels, total_loss, energy_loss, gradient_loss, charge_loss, num_atoms = self.sess.run([self.total_energy,
				self.energy_pl, self.gradients, self.gradient_labels, self.charges, self.charge_labels,
				self.total_loss, self.energy_loss, self.gradient_loss, self.charge_loss, self.num_atoms_pl],  feed_dict=feed_dict)
				test_charge_loss += charge_loss
				test_charge_labels.append(charge_labels)
				test_charge_outputs.append(charges)
			else:
				total_energies, energy_labels, gradients, gradient_labels, total_loss, energy_loss, gradient_loss, num_atoms = self.sess.run([self.total_energy,
				self.energy_pl, self.gradients, self.gradient_labels, self.total_loss, self.energy_loss,
				self.gradient_loss, self.num_atoms_pl],  feed_dict=feed_dict)
			test_loss += total_loss
			test_energy_loss += energy_loss
			test_gradient_loss += gradient_loss
			test_energy_labels.append(energy_labels)
			test_energy_outputs.append(total_energies)
			test_force_labels.append(-1.0 * gradient_labels)
			test_force_outputs.append(-1.0 * gradients)
			num_atoms_epoch += np.sum(num_atoms)
			num_batches += 1
		test_loss /= num_batches
		test_energy_loss /= num_batches
		test_gradient_loss /= num_batches
		test_charge_loss /= num_batches
		test_energy_labels = np.concatenate(test_energy_labels)
		test_energy_outputs = np.concatenate(test_energy_outputs)
		test_energy_errors = test_energy_labels - test_energy_outputs
		test_force_labels = np.concatenate(test_force_labels)
		test_force_outputs = np.concatenate(test_force_outputs)
		test_force_errors = test_force_labels - test_force_outputs
		duration = time.time() - start_time
		for i in [random.randint(0, num_batches * self.batch_size - 1) for _ in range(10)]:
			LOGGER.info("Energy label: %12.8f  Energy output: %12.8f", test_energy_labels[i], test_energy_outputs[i])
		for i in [random.randint(0, num_atoms_epoch - 1) for _ in range(10)]:
			LOGGER.info("Forces label: %s  Forces output: %s", test_force_labels[i], test_force_outputs[i])
		if self.train_charges:
			test_charge_labels = np.concatenate(test_charge_labels)
			test_charge_outputs = np.concatenate(test_charge_outputs)
			test_charge_errors = test_charge_labels - test_charge_outputs
			for i in [random.randint(0, num_atoms_epoch - 1) for _ in range(10)]:
				LOGGER.info("Charge label: %11.8f  Charge output: %11.8f", test_charge_labels[i], test_charge_outputs[i])
			LOGGER.info("MAE  Energy: %11.8f  Forces: %11.8f  Charges %11.8f", np.mean(np.abs(test_energy_errors)),
			np.mean(np.abs(test_force_errors)), np.mean(np.abs(test_charge_errors)))
			LOGGER.info("MSE  Energy: %11.8f  Forces: %11.8f  Charges %11.8f", np.mean(test_energy_errors),
			np.mean(test_force_errors), np.mean(test_charge_errors))
			LOGGER.info("RMSE Energy: %11.8f  Forces: %11.8f  Charges %11.8f", np.sqrt(np.mean(np.square(test_energy_errors))),
			np.sqrt(np.mean(np.square(test_force_errors))), np.sqrt(np.mean(np.square(test_charge_errors))))
		else:
			LOGGER.info("MAE  Energy: %11.8f  Forces: %11.8f", np.mean(np.abs(test_energy_errors)),
			np.mean(np.abs(test_force_errors)))
			LOGGER.info("MSE  Energy: %11.8f  Forces: %11.8f", np.mean(test_energy_errors),
			np.mean(test_force_errors))
			LOGGER.info("RMSE Energy: %11.8f  Forces: %11.8f", np.sqrt(np.mean(np.square(test_energy_errors))),
			np.sqrt(np.mean(np.square(test_force_errors))))
		self.print_epoch(step, duration, test_loss, test_energy_loss, test_gradient_loss, test_charge_loss, testing=True)
		return test_loss

	def compute_normalization(self):
		if self.train_charges:
			self.charge_mean = np.zeros((self.mol_set.max_atomic_num()+1))
			self.charge_std = np.zeros((self.mol_set.max_atomic_num()+1))
			for element in self.elements:
				element_idxs = np.where(np.equal(self.Z_data, element))
				element_charges = self.charges_data[element_idxs]
				self.charge_mean[element] = np.mean(element_charges)
				self.charge_std[element] = np.std(element_charges)
		energies = self.energy_data - np.sum(self.energy_fit[self.Z_data], axis=1)
		self.energy_mean = np.mean(energies)
		self.energy_stddev = np.std(energies)
		self.embed_shape = self.element_codes.shape[1] * (self.radial_rs.shape[0] + self.angular_rs.shape[0] * self.theta_s.shape[0])
		self.label_shape = self.energy_mean.shape
		return

	def train_prepare(self, restart=False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
			self.Zs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms])
			self.nn_pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None])
			self.nn_triples_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None, 2])
			self.coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
			self.energy_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size])
			self.gradients_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
			self.charges_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms])

			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=True, dtype=self.tf_precision, name="element_codes")
			self.element_codepairs = tf.Variable(self.element_codepairs, trainable=True, dtype=self.tf_precision, name="element_codepairs")
			self.codepair_idx = tf.Variable(self.codepair_idx, trainable=False, dtype=tf.int32)
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			charge_fit = tf.Variable(self.charge_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_stddev = tf.Variable(self.energy_stddev, trainable=False, dtype = self.tf_precision)
			charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
			charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl, self.element_codes,
					self.element_codepairs, self.codepair_idx, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.mol_nn_energy = tf.reduce_sum(atom_nn_energy, axis=1) * energy_stddev
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy

				dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, atom_nn_charges, self.coulomb_pairs_pl)
				self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
				# self.mol_coulomb_energy = tf.reshape(tf.reduce_sum(atom_coulomb_energy, axis=1), [self.batch_size])
				self.total_energy += self.mol_coulomb_energy
				self.charges = tf.gather_nd(atom_nn_charges, padding_mask)
				self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
				self.charge_loss = self.loss_op(self.charges - self.charge_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
				tf.summary.scalar("charge_loss", self.charge_loss)
				tf.add_to_collection('total_loss', self.charge_loss)
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, self.Zs_pl, self.num_atoms_pl)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
					self.charges = tf.gather_nd(atom_charges, padding_mask)
					self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
					self.charge_loss = self.loss_op(self.charges - self.charge_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
					tf.summary.scalar("charge_loss", self.charge_loss)
					tf.add_to_collection('total_loss', self.charge_loss)
			self.energy_loss = 100 * self.loss_op(self.total_energy - self.energy_pl) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
			tf.summary.scalar("energy_loss", self.energy_loss)
			tf.add_to_collection('total_loss', self.energy_loss)
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
				self.gradient_labels = tf.gather_nd(self.gradients_pl, padding_mask)
				self.gradient_loss = self.loss_op(self.gradients - self.gradient_labels) / (3 * tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision))
				if self.train_gradients:
					tf.add_to_collection('total_loss', self.gradient_loss)
					tf.summary.scalar("gradient_loss", self.gradient_loss)
			self.total_loss = tf.add_n(tf.get_collection('total_loss'))
			tf.summary.scalar('total_loss', self.total_loss)

			self.train_op = self.optimizer(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
			if restart:
				self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
			else:
				init = tf.global_variables_initializer()
				self.sess.run(init)
			if self.profiling:
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
		return

	def print_epoch(self, step, duration, loss, energy_loss, gradient_loss, charge_loss, testing=False):
		if testing:
			LOGGER.info("step: %5d  duration: %.3f  test loss: %.10f  energy loss: %.10f  gradient loss: %.10f  charge loss: %.10f",
			step, duration, loss, energy_loss, gradient_loss, charge_loss)
		else:
			LOGGER.info("step: %5d  duration: %.3f  train loss: %.10f  energy loss: %.10f  gradient loss: %.10f  charge loss: %.10f",
			step, duration, loss, energy_loss, gradient_loss, charge_loss)
		return

	def evaluate_set(self):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		self.assign_activation()
		self.reload_set()
		self.load_data()
		self.train_prepare(restart=True)
		labels, preds, idxs = [], [], []
		for ministep in range (0, int(0.05 * self.num_train_cases/self.batch_size)):
			batch_data = self.get_train_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			total_loss, energy_loss, total_energy, energy_label = self.sess.run([self.total_loss,
				self.energy_loss, self.total_energy, self.energy_pl], feed_dict=feed_dict)
			labels.append(energy_label)
			preds.append(total_energy)
			idxs.append(self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer])
		labels = np.concatenate(labels)
		preds = np.concatenate(preds)
		idxs = np.concatenate(idxs)
		return labels, preds, idxs

	def evaluate_mol(self, mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.batch_size = 1
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.train_prepare(restart=True)
		xyzs_feed = mol.coords
		Zs_feed = mol.atoms
		nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		feed_dict={self.xyzs_pl:xyzs_feed, self.Zs_pl:Zs_feed, self.nn_pairs_pl:nn_pairs, self.nn_triples_pl:nn_triples}
		energy, gradients = self.sess.run([self.total_energy, self.xyz_grad], feed_dict=feed_dict)
		return energy, -gradients
