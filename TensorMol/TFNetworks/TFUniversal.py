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
	The 0.2 model chemistry.
	"""
	def __init__(self, mol_set_name=None, max_num_atoms=None, name=None):
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
		self.codes_shape = self.element_codes.shape[1]
		self.element_codepairs = np.zeros((int(self.element_codes.shape[0]*(self.element_codes.shape[0]+1)/2), self.element_codes.shape[1]))
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
			self.path = PARAMS["networks_directory"]
			self.network_directory = PARAMS["networks_directory"]+self.name
			#self.max_num_atoms = max_num_atoms if max_num_atoms else self.mol_set.MaxNAtom()
			LOGGER.info("Reloaded network from %s", self.network_directory)
			return

		#Data parameters
		self.mol_set_name = mol_set_name
		self.mol_set = MSet(self.mol_set_name)
		self.mol_set.Load()
		self.elements = self.mol_set.AtomTypes()
		self.max_num_atoms = max_num_atoms if max_num_atoms else self.mol_set.MaxNAtom()
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
			elif self.activation_function_type == "gaussian":
				self.activation_function = self.gaussian_activation
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

	def gaussian_activation(self, x):
		return tf.exp(-tf.square(x))

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
		if self.train_charges:
			self.charges_data = np.zeros((self.num_molecules, self.max_num_atoms), dtype = np.float64)
		self.num_atoms_data = np.zeros((self.num_molecules), dtype = np.int32)
		self.energy_data = np.zeros((self.num_molecules), dtype = np.float64)
		self.gradient_data = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(self.mol_set.mols):
			self.xyz_data[i][:mol.NAtoms()] = mol.coords
			self.Z_data[i][:mol.NAtoms()] = mol.atoms
			self.energy_data[i] = mol.properties["energy"]
			self.gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			self.num_atoms_data[i] = mol.NAtoms()
			if self.train_charges:
				self.charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
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

	def get_train_batch(self):
		if self.train_pointer + self.batch_size >= self.num_train_cases:
			np.random.shuffle(self.train_idxs)
			self.train_pointer = 0
		self.train_pointer += self.batch_size
		batch_xyzs = self.xyz_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]]
		batch_Zs = self.Z_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]]
		nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
		batch_data = []
		batch_data.append(batch_xyzs)
		batch_data.append(batch_Zs)
		batch_data.append(nn_pairs)
		batch_data.append(nn_triples)
		batch_data.append(coulomb_pairs)
		batch_data.append(self.num_atoms_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		batch_data.append(self.energy_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		batch_data.append(self.gradient_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		if self.train_charges:
			batch_data.append(self.charges_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		return batch_data

	def get_test_batch(self):
		if self.test_pointer + self.batch_size >= self.num_test_cases:
			self.test_pointer = 0
		self.test_pointer += self.batch_size
		batch_xyzs = self.xyz_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]]
		batch_Zs = self.Z_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]]
		nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
		batch_data = []
		batch_data.append(batch_xyzs)
		batch_data.append(batch_Zs)
		batch_data.append(nn_pairs)
		batch_data.append(nn_triples)
		batch_data.append(coulomb_pairs)
		batch_data.append(self.num_atoms_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]])
		batch_data.append(self.energy_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]])
		batch_data.append(self.gradient_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]])
		if self.train_charges:
			batch_data.append(self.charges_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]])
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
					self.num_atoms_pl, self.energy_pl, self.gradients_pl]
		if self.train_charges:
			pl_list.append(self.charges_pl)
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
			code_kernel1 = tf.get_variable(name="CodeKernel1", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
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
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
				outputs = tf.squeeze(tf.matmul(activations, weights) + biases, axis=1)
				variables.append(weights)
				variables.append(biases)
				atom_nn_energy = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
		return atom_nn_energy, variables

	def charge_inference(self, embed, atom_codes, indices):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		variables=[]
		with tf.variable_scope("charge_network", reuse=tf.AUTO_REUSE):
			code_kernel1 = tf.get_variable(name="CodeKernel", shape=(self.codes_shape, self.codes_shape),dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(self.codes_shape, self.codes_shape),dtype=self.tf_precision)
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
				atom_nn_charges = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
		return atom_nn_charges, variables

	def charge_equalization(self, atom_nn_charges, num_atoms, Zs):
		excess_charge = tf.reduce_sum(atom_nn_charges, axis=1)
		atom_nn_charges -= tf.expand_dims(excess_charge / tf.cast(num_atoms, eval(PARAMS["tf_prec"])), axis=-1)
		mask = tf.where(tf.equal(Zs, 0), tf.zeros_like(Zs, dtype=eval(PARAMS["tf_prec"])),
				tf.ones_like(Zs, dtype=eval(PARAMS["tf_prec"])))
		atom_nn_charges = atom_nn_charges * mask
		return atom_nn_charges

	def alchem_charge_equalization(self, atom_nn_charges, num_alchem_atoms, alchem_switch):
		excess_charge = tf.reduce_sum(atom_nn_charges, axis=1)
		atom_nn_charges -= (excess_charge / num_alchem_atoms) * alchem_switch
		return atom_nn_charges

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

	def alchem_gather_coulomb(self, xyzs, atom_charges, pairs):
		padding_mask = tf.where(tf.reduce_any(tf.not_equal(pairs, -1), axis=-1))
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
		srange_inner = tf.constant(6.0*1.889725989, dtype=tf.float64)
		srange_outer = tf.constant(9.0*1.889725989, dtype=tf.float64)
		lrange_inner = tf.constant(13.0*1.889725989, dtype=tf.float64)
		lrange_outer = tf.constant(15.0*1.889725989, dtype=tf.float64)
		a, b, c, d, e, f, g, h = -43.568, 15.9138, -2.42286, 0.203849, -0.0102346, 0.000306595, -5.0738e-6, 3.57816e-8
		dist = tf.norm(dxyzs+1.e-16, axis=-1)
		dist *= 1.889725989
		dist = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) * srange_inner, dist)
		dist = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) * lrange_outer, dist)
		dist2 = dist * dist
		dist3 = dist2 * dist
		dist4 = dist3 * dist
		dist5 = dist4 * dist
		dist6 = dist5 * dist
		dist7 = dist6 * dist
		kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
		mrange_energy = tf.reduce_sum(kern * q1q2, axis=1)
		lrange_energy = tf.reduce_sum(q1q2, axis=1) / lrange_outer
		coulomb_energy = (mrange_energy - lrange_energy) / 2.0
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
		for ministep in range (0, int(0.1 * Ncase_train/self.batch_size)):
			batch_data = self.get_train_batch()
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
			batch_data = self.get_test_batch()
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
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codepairs(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl, self.element_codes,
					self.element_codepairs, self.codepair_idx, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.mol_nn_energy = tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
					dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
					self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
					self.charge_loss = 0.1 * self.loss_op(self.charges - self.charge_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
					tf.summary.scalar("charge_loss", self.charge_loss)
					tf.add_to_collection('total_loss', self.charge_loss)
			self.energy_loss = 100.0 * self.loss_op(self.total_energy - self.energy_pl) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
			tf.summary.scalar("energy_loss", self.energy_loss)
			tf.add_to_collection('total_loss', self.energy_loss)
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
				self.gradient_labels = tf.gather_nd(self.gradients_pl, padding_mask)
				self.gradient_loss = 10.0 * self.loss_op(self.gradients - self.gradient_labels) / (3 * tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision))
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

	def alchem_prepare(self):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[None, self.max_num_atoms, 3])
			self.Zs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms])
			self.nn_pairs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None])
			self.nn_triples_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None, 2])
			self.coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[None])
			self.delta_pl = tf.placeholder(self.tf_precision, shape=[1])

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
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			self.padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			self.alchem_padding_mask = tf.where(tf.reduce_any(tf.not_equal(self.Zs_pl, 0), axis=0, keepdims=True))
			_, self.max_atom_idx = tf.nn.top_k(self.num_atoms_pl)
			self.alchem_xyzs = tf.gather(self.xyzs_pl, self.max_atom_idx)
			self.alchem_switch = tf.where(tf.not_equal(self.Zs_pl, 0), tf.stack([tf.tile(1.0 - self.delta_pl,
								[self.max_num_atoms]), tf.tile(self.delta_pl, [self.max_num_atoms])]),
								tf.zeros_like(self.Zs_pl, dtype=eval(PARAMS["tf_prec"])))
			self.embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl, self.element_codes,
					self.element_codepairs, self.codepair_idx, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			self.reconst_embed = tf.scatter_nd(self.padding_mask, self.embed, [tf.cast(tf.shape(self.Zs_pl)[0], tf.int64), self.max_num_atoms, self.element_codes.shape[1], (self.radial_rs.shape[0] + self.angular_rs.shape[0] * self.theta_s.shape[0])])
			self.alchem_embed = tf.reduce_sum(tf.stack([self.reconst_embed[0] * (1.0 - self.delta_pl), self.reconst_embed[1] * self.delta_pl], axis=0), axis=0)
			self.atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, self.padding_mask))
			self.reconst_atom_codes = tf.scatter_nd(self.padding_mask, self.atom_codes, [tf.cast(tf.shape(self.Zs_pl)[0], tf.int64), self.max_num_atoms, 4])
			self.alchem_atom_codes = tf.reduce_sum(tf.stack([self.reconst_atom_codes[0] * (1.0 - self.delta_pl), self.reconst_atom_codes[1] * self.delta_pl], axis=0), axis=0)
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(self.alchem_embed, self.alchem_atom_codes, self.alchem_padding_mask)
				self.atom_nn_energy *= tf.reduce_sum(self.alchem_switch, axis=0)
				self.mol_nn_energy = tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_stddev
				self.mol_energy_fit = tf.reduce_sum(tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl) * self.alchem_switch, axis=0), axis=0)
				self.mol_nn_energy += self.mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					self.atom_nn_charges, charge_variables = self.charge_inference(self.alchem_embed, self.atom_codes, self.alchem_padding_mask)
					self.atom_charge_mean, self.atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_charge_mean *= self.alchem_switch
					self.atom_charge_std *= self.alchem_switch
					self.atom_charge_mean = tf.reduce_sum(self.atom_charge_mean, axis=0)
					self.atom_charge_std = tf.reduce_sum(self.atom_charge_std, axis=0)
					self.atom_nn_charges = (self.atom_nn_charges * self.atom_charge_std) + self.atom_charge_mean
					self.num_alchem_atoms = tf.reduce_sum(self.alchem_switch)
					self.atom_nn_charges = self.alchem_charge_equalization(self.atom_nn_charges, self.num_alchem_atoms, tf.reduce_sum(self.alchem_switch, axis=0))
					dxyzs, q1q2, scatter_coulomb = self.alchem_gather_coulomb(self.alchem_xyzs, self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
			with tf.name_scope('gradients'):
				self.gradients = tf.reduce_sum(tf.gradients(self.total_energy, self.xyzs_pl)[0], axis=0)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def element_opt_prepare(self):
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
			self.atom_codes_pl = tf.placeholder(self.tf_precision, shape=[None, 4])
			self.atom_codepairs_pl = tf.placeholder(self.tf_precision, shape=[None, 55, 4])
			self.replace_bool_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms])
			self.replace_atom_idx = tf.cast(tf.where(tf.equal(self.replace_bool_pl, 1)), tf.int32)
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
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes_v2(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl,
					self.nn_triples_pl, self.element_codes, self.element_codepairs, self.codepair_idx,
					radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta,
					self.replace_atom_idx, self.atom_codes_pl, self.atom_codepairs_pl)
			atom_codes = tf.gather(self.element_codes, self.Zs_pl)
			atom_codes = tf.where(tf.equal(tf.tile(tf.expand_dims(self.replace_bool_pl, axis=-1), [1, 1, 4]), 1),
				tf.tile(tf.reshape(self.atom_codes_pl, [1, 1, 4]), [1, self.max_num_atoms, 1]), atom_codes)
			atom_codes = tf.gather_nd(atom_codes, padding_mask)
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.mol_nn_energy = tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_stddev
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
					dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
			with tf.name_scope('gradients'):
				self.xyz_grad, self.atom_codes_grad, self.atom_codepairs_grad = tf.gradients(self.total_energy,
										[self.xyzs_pl, self.atom_codes_pl, self.atom_codepairs_pl])
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)

			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def print_epoch(self, step, duration, loss, energy_loss, gradient_loss, charge_loss, testing=False):
		if testing:
			LOGGER.info("step: %5d  duration: %.3f  test loss: %.10f  energy loss: %.10f  gradient loss: %.10f  charge loss: %.10f",
			step, duration, loss, energy_loss, gradient_loss, charge_loss)
		else:
			LOGGER.info("step: %5d  duration: %.3f  train loss: %.10f  energy loss: %.10f  gradient loss: %.10f  charge loss: %.10f",
			step, duration, loss, energy_loss, gradient_loss, charge_loss)
		return

	def get_eval_batch(self, batch_size):
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

	def evaluate_set(self, mset):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		self.assign_activation()
		self.max_num_atoms = mset.MaxNAtom()
		self.batch_size = 200
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			energy_data[i] = mol.properties["energy"]
			gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0
		self.train_prepare(restart=True)
		energy_true, energy_pred = [], []
		gradients_true, gradient_preds = [], []
		charges_true, charge_preds = [], []
		for ministep in range(int(num_mols / self.batch_size)):
			eval_pointer += self.batch_size
			batch_xyzs = xyz_data[eval_pointer - self.batch_size:eval_pointer]
			batch_Zs = Z_data[eval_pointer - self.batch_size:eval_pointer]
			nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
			batch_data = []
			batch_data.append(batch_xyzs)
			batch_data.append(batch_Zs)
			batch_data.append(nn_pairs)
			batch_data.append(nn_triples)
			batch_data.append(coulomb_pairs)
			batch_data.append(num_atoms_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(energy_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(gradient_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(charges_data[eval_pointer - self.batch_size:eval_pointer])
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_charges:
				total_energy, energy_label, gradients, gradient_labels, charges, charge_labels = self.sess.run([self.total_energy,
					self.energy_pl, self.gradients, self.gradient_labels, self.charges, self.charge_labels], feed_dict=feed_dict)
				charges_true.append(charge_labels)
				charge_preds.append(charges)
			else:
				total_energy, energy_label, gradients, gradient_labels = self.sess.run([self.total_energy,
					self.energy_pl, self.gradients, self.gradient_labels], feed_dict=feed_dict)
			energy_true.append(energy_label)
			energy_pred.append(total_energy)
			gradients_true.append(gradient_labels)
			gradient_preds.append(gradients)
		energy_true = np.concatenate(energy_true)
		energy_pred = np.concatenate(energy_pred)
		gradients_true = np.concatenate(gradients_true)
		gradient_preds = np.concatenate(gradient_preds)
		energy_errors = energy_true - energy_pred
		gradient_errors = gradients_true - gradient_preds
		if self.train_charges:
			charges_true = np.concatenate(charges_true)
			charge_preds = np.concatenate(charge_preds)
			charge_errors = charges_true - charge_preds
			return energy_errors, gradient_errors, charge_errors
		else:
			return energy_errors, gradient_errors

	def evaluate_mol(self, mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.train_prepare(restart=True)
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		xyz_data[0][:mol.NAtoms()] = mol.coords
		Z_data[0][:mol.NAtoms()] = mol.atoms
		num_atoms_data[0] = mol.NAtoms()
		nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 19.0, self.max_num_atoms, False, False)
		feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
					self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs, self.num_atoms_pl:num_atoms_data}
		energy, atme, nne, ce, gradients, charges = self.sess.run([self.total_energy, self.atom_nn_energy, self.mol_nn_energy, self.mol_coulomb_energy, self.gradients, self.atom_nn_charges], feed_dict=feed_dict)
		return energy, atme, nne, ce, -gradients, charges

	def evaluate_alchem_mol(self, mols):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = max([mol.NAtoms() for mol in mols])
			self.batch_size = 1
			self.alchem_prepare(restart=True)
		xyz_data = np.zeros((len(mols), self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((len(mols), self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((len(mols)), dtype = np.int32)
		def alchem_energy_force(mols, delta, return_forces=True):
			for i, mol in enumerate(mols):
				xyz_data[i][:mol.NAtoms()] = mols[i].coords
				Z_data[i][:mol.NAtoms()] = mols[i].atoms
				num_atoms_data[i] = mol.NAtoms()
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(xyz_data[np.argmax(num_atoms_data):np.argmax(num_atoms_data)+1],
							Z_data[np.argmax(num_atoms_data):np.argmax(num_atoms_data)+1], 19.0, self.max_num_atoms, False, False)
			delta = np.array(delta).reshape(1)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs,
						self.num_atoms_pl:num_atoms_data, self.delta_pl:delta}
			energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
			return energy[0], -gradients
		return alchem_energy_force

	def get_element_opt_function(self, mol, replace_idxs):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.element_opt_prepare()
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		replace_idx_data = np.zeros((num_mols, self.max_num_atoms), dtype=np.int32)
		atom_codes_data = np.zeros((len(replace_idxs), 4), dtype=np.float64)
		atom_codepairs_data = np.zeros((len(replace_idxs), 55, 4), dtype=np.float64)
		num_atoms_data[0] = mol.NAtoms()
		Z_data[0][:mol.NAtoms()] = mol.atoms
		for i in range(len(replace_idxs)):
			replace_idx_data[replace_idxs[i][0], replace_idxs[i][1]] = 1
		atom_codes = self.sess.run(self.element_codes)
		def EF(xyz, atom_codes, atom_codepairs, DoForce=True):
			xyz_data[0][:mol.NAtoms()] = xyz
			atom_codes_data[0] = atom_codes
			atom_codepairs_data[0] = atom_codepairs
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			# coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 19.0, self.max_num_atoms, False, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.num_atoms_pl:num_atoms_data,
						self.replace_bool_pl:replace_idx_data, self.atom_codes_pl:atom_codes_data,
						self.atom_codepairs_pl:atom_codepairs_data}

			energy, gradients, codes_gradient, codepairs_gradient = self.sess.run([self.total_energy,
					self.gradients, self.atom_codes_grad, self.atom_codepairs_grad], feed_dict=feed_dict)
			return energy[0], -gradients, -codes_gradient, -codepairs_gradient
		element_codes, element_codepairs, codepair_idx = self.sess.run([self.element_codes, self.element_codepairs, self.codepair_idx])
		original_codes = element_codes[mol.atoms[replace_idx_data[:,1]]]
		original_codepair_gather = codepair_idx[mol.atoms[replace_idx_data[:,1]]]
		original_codepairs = element_codepairs[original_codepair_gather]
		return EF, original_codes, original_codepairs


	def get_energy_force_function(self,mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.train_prepare(restart=True)
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		num_atoms_data[0] = mol.NAtoms()
		Z_data[0][:mol.NAtoms()] = mol.atoms
		def EF(xyz_, DoForce=True):
			xyz_data[0][:mol.NAtoms()] = xyz_
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 15.0, self.max_num_atoms, True, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs, self.num_atoms_pl:num_atoms_data}
			if (DoForce):
				energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
			# print(-JOULEPERHARTREE*gradients)
				return energy[0], -JOULEPERHARTREE*gradients
			else:
				energy = self.sess.run(self.total_energy, feed_dict=feed_dict)
				return energy[0]
		return EF

	def get_charge_function(self,mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.train_prepare(restart=True)
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		num_atoms_data[0] = mol.NAtoms()
		Z_data[0][:mol.NAtoms()] = mol.atoms
		def QF(xyz_):
			xyz_data[0][:mol.NAtoms()] = xyz_
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.num_atoms_pl:num_atoms_data}
			charges = self.sess.run(self.charges, feed_dict=feed_dict)
			return charges
		return QF

	def GetBatchedEnergyForceRoutine(self,mset):
		self.assign_activation()
		self.max_num_atoms = mset.MaxNAtom()
		self.batch_size = len(mset.mols)
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0

		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.train_prepare(restart=True)

		def EF(xyzs_, DoForce=True):
			xyz_data[:,:mol.NAtoms(),:] = xyzs_.copy()
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 19.0, self.max_num_atoms, False, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs, self.num_atoms_pl:num_atoms_data}
			energy, gradients, charges = self.sess.run([self.total_energy, self.gradients, self.atom_nn_charges], feed_dict=feed_dict)
			if (DoForce):
				return energy, -JOULEPERHARTREE*gradients
			else:
				return energy
		return EF

class UniversalNetwork_v2(UniversalNetwork):
	"""
	The 0.2 model chemistry.
	"""
	def __init__(self, mol_set_name=None, max_num_atoms=None, name=None):
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
		self.element_codes = ELEMENTCODES6
		self.codes_shape = self.element_codes.shape[1]
		self.assign_activation()

		#Reloads a previous network if name variable is not None
		if name != None:
			self.name = name
			self.load_network()
			self.path = PARAMS["networks_directory"]
			self.network_directory = PARAMS["networks_directory"]+self.name
			#self.max_num_atoms = max_num_atoms if max_num_atoms else self.mol_set.MaxNAtom()
			LOGGER.info("Reloaded network from %s", self.network_directory)
			return

		#Data parameters
		self.mol_set_name = mol_set_name
		self.mol_set = MSet(self.mol_set_name)
		self.mol_set.Load()
		self.elements = self.mol_set.AtomTypes()
		self.max_num_atoms = max_num_atoms if max_num_atoms else self.mol_set.MaxNAtom()
		self.num_molecules = len(self.mol_set.mols)
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
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std)
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
					dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
					self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
					self.charge_loss = 0.1 * self.loss_op(self.charges - self.charge_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
					tf.summary.scalar("charge_loss", self.charge_loss)
					tf.add_to_collection('total_loss', self.charge_loss)
			self.energy_loss = 100.0 * self.loss_op(self.total_energy - self.energy_pl) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
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

	def eval_prepare(self):
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
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.atom_nn_energy_tmp = self.atom_nn_energy * energy_std
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std) + energy_mean
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
					dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
					self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
				self.gradient_labels = tf.gather_nd(self.gradients_pl, padding_mask)

			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def alchem_prepare(self):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[None, self.max_num_atoms, 3])
			self.Zs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms])
			self.nn_pairs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None])
			self.nn_triples_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None, 2])
			self.coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[None])
			self.delta_pl = tf.placeholder(self.tf_precision, shape=[1])

			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=True, dtype=self.tf_precision, name="element_codes")
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			alchem_padding_mask = tf.where(tf.reduce_any(tf.not_equal(self.Zs_pl, 0), axis=0, keepdims=True))
			_, max_atom_idx = tf.nn.top_k(self.num_atoms_pl)
			self.alchem_xyzs = tf.gather(self.xyzs_pl, max_atom_idx)
			self.alchem_switch = tf.where(tf.not_equal(self.Zs_pl, 0), tf.stack([tf.tile(1.0 - self.delta_pl,
								[self.max_num_atoms]), tf.tile(self.delta_pl, [self.max_num_atoms])]),
								tf.zeros_like(self.Zs_pl, dtype=eval(PARAMS["tf_prec"])))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			reconst_embed = tf.scatter_nd(padding_mask, embed, [tf.cast(tf.shape(self.Zs_pl)[0], tf.int64),
				self.max_num_atoms, self.codes_shape, int(self.embed_shape / self.codes_shape)])
			alchem_embed = tf.reduce_sum(tf.stack([reconst_embed[0] * (1.0 - self.delta_pl),
				reconst_embed[1] * self.delta_pl], axis=0), axis=0)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			reconst_atom_codes = tf.scatter_nd(padding_mask, atom_codes,
				[tf.cast(tf.shape(self.Zs_pl)[0], tf.int64), self.max_num_atoms, self.codes_shape])
			alchem_atom_codes = tf.reduce_sum(tf.stack([reconst_atom_codes[0] * (1.0 - self.delta_pl),
				reconst_atom_codes[1] * self.delta_pl], axis=0), axis=0)
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(alchem_embed,
					alchem_atom_codes, alchem_padding_mask)
				self.atom_nn_energy *= tf.reduce_sum(self.alchem_switch, axis=0)
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std) + energy_mean
				mol_energy_fit = tf.reduce_sum(tf.reduce_sum(tf.gather(energy_fit,
					self.Zs_pl) * self.alchem_switch, axis=0), axis=0)
				# self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(alchem_embed,
						alchem_atom_codes, alchem_padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					atom_charge_mean *= self.alchem_switch
					atom_charge_std *= self.alchem_switch
					atom_charge_mean = tf.reduce_sum(atom_charge_mean, axis=0)
					atom_charge_std = tf.reduce_sum(atom_charge_std, axis=0)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					num_alchem_atoms = tf.reduce_sum(self.alchem_switch)
					self.atom_nn_charges = self.alchem_charge_equalization(self.atom_nn_charges,
						num_alchem_atoms, tf.reduce_sum(self.alchem_switch, axis=0))
					dxyzs, q1q2, scatter_coulomb = self.alchem_gather_coulomb(self.alchem_xyzs,
						self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
			with tf.name_scope('gradients'):
				self.gradients = tf.reduce_sum(tf.gradients(self.total_energy, self.xyzs_pl)[0], axis=0)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def element_opt_prepare(self):
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
			self.replace_idx_pl = tf.placeholder(tf.int32, shape=[1, 2])

			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=True,
				dtype=self.tf_precision, name="element_codes")
			elements = tf.constant(self.elements, dtype = tf.int32)
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			self.replace_scalars = tf.Variable([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], trainable=True,
				dtype=self.tf_precision, name="replace_scalars")
			self.lagrange_mult1 = tf.Variable(1.0, trainable=True, dtype=self.tf_precision,
				name="lagrange_mult1")
			self.lagrange_mult2 = tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
				1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], trainable=True,
				dtype=self.tf_precision, name="lagrange_mult2")
			self.lagrange_mult3 = tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
				1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], trainable=True,
				dtype=self.tf_precision, name="lagrange_mult3")
			self.zero_bound = tf.Variable(self.replace_scalars, trainable=True,
				dtype=self.tf_precision, name="zero_bound")
			self.one_bound = tf.Variable(1.0 - self.replace_scalars, trainable=True,
				dtype=self.tf_precision, name="one_bound")

			self.replace_codes = tf.reduce_sum(tf.expand_dims(self.replace_scalars, axis=-1)
				* tf.gather(self.element_codes, elements), axis=0)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes_replace(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl,
					self.nn_triples_pl, self.element_codes, radial_gauss, radial_cutoff, angular_gauss,
					thetas, angular_cutoff, zeta, eta, self.replace_idx_pl, self.replace_codes)
			atom_codes = tf.gather(self.element_codes, self.Zs_pl)
			atom_idx = tf.range(self.num_atoms_pl[0])
			replace_broadcast = tf.where(tf.equal(atom_idx, self.replace_idx_pl[0,1]),
				tf.ones_like(atom_idx, dtype=self.tf_precision),
				tf.zeros_like(atom_idx, dtype=self.tf_precision))
			codes_broadcast = tf.where(tf.equal(atom_idx, self.replace_idx_pl[0,1]),
				tf.zeros_like(atom_idx, dtype=self.tf_precision),
				tf.ones_like(atom_idx, dtype=self.tf_precision))
			atom_codes = (tf.expand_dims(replace_broadcast, axis=-1) * self.replace_codes
				+ tf.expand_dims(codes_broadcast, axis=-1) * atom_codes)
			atom_codes = tf.gather_nd(atom_codes, padding_mask)
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std) + energy_mean
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				# self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
					dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
			self.aux_func = self.lagrange_mult1 * (tf.reduce_sum(self.replace_scalars) - 1.0)
			self.zero_bound_func = self.lagrange_mult2 * (self.replace_scalars - tf.square(self.zero_bound))
			self.one_bound_func = self.lagrange_mult3 * (1.0 - self.replace_scalars - tf.square(self.one_bound))
			self.lagrangian = self.total_energy + self.aux_func + self.zero_bound_func + self.one_bound_func
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
				# self.scalars_energy_grad = tf.gradients(self.total_energy, self.replace_scalars)[0]
				# self.scalars_aux_grad, self.lagrange_grad = tf.gradients(self.aux_func,
				# 	[self.replace_scalars, self.lagrange_mult])
				# self.scalars_energy_loss = self.loss_op(self.scalars_energy_grad)
				# self.scalars_aux_loss = self.loss_op(self.scalars_aux_grad)
				# self.lagrange_loss = self.loss_op(self.lagrange_grad)
				(self.scalars_grad, self.lagrange_grad1, self.lagrange_grad2, self.lagrange_grad3,
					self.zero_grad, self.one_grad) = tf.gradients(self.lagrangian,
					[self.replace_scalars, self.lagrange_mult1, self.lagrange_mult2, self.lagrange_mult3,
					self.zero_bound, self.one_bound])
				self.scalars_loss = self.loss_op(self.scalars_grad)
				self.lagrange1_loss = self.loss_op(self.lagrange_grad1)
				self.lagrange2_loss = self.loss_op(self.lagrange_grad2)
				self.lagrange3_loss = self.loss_op(self.lagrange_grad3)
				self.zero_loss = self.loss_op(self.zero_grad)
				self.one_loss = self.loss_op(self.one_grad)
				self.total_loss = (self.scalars_loss + (self.lagrange1_loss
					+ self.lagrange2_loss + self.lagrange3_loss + self.zero_loss + self.one_loss))

			self.minimize_op = self.optimizer(self.total_loss, 10.0 * self.learning_rate,
				self.momentum, [self.replace_scalars, self.lagrange_mult1, self.lagrange_mult2,
				self.lagrange_mult3, self.zero_bound, self.one_bound])

			vars_in_checkpoint = tf.train.list_variables(self.network_directory)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
			self.sess.run(tf.variables_initializer(all_variables))
			restore_vars = [v for v in all_variables if ("lagrange_mult" not in v.name) and ("replace_scalars" not in v.name) and ("bound" not in v.name)]
			self.saver = tf.train.Saver(var_list=restore_vars)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def compute_normalization(self):
		elements = self.mol_set.AtomTypes().tolist()
		avg_energy = {element:0. for element in elements}
		num_elements = len(avg_energy)
		num_elements_dense = max(elements)+1
		regres_energies = np.zeros(self.num_molecules)
		# Use dense zero-padded arrays to avoid index logic.
		stoichs = np.zeros((self.num_molecules, num_elements_dense))

		srange_inner = 4.0*1.889725989
		srange_outer = 6.5*1.889725989
		lrange_inner = 13.0*1.889725989
		lrange_outer = 15.0*1.889725989
		a, b, c, d, e, f, g, h = -9.83315, 4.49307, -0.784438, 0.0747019, -0.00419095, 0.000138593, -2.50374e-6, 1.90818e-8

		for i, mol in enumerate(self.mol_set.mols):
			unique, counts = np.unique(mol.atoms, return_counts=True)
			stoich = np.zeros(num_elements_dense)
			for j in range(len(unique)):
				stoich[unique[j]] = counts[j]
			stoichs[i] += stoich
			if self.train_charges:
				mol.BuildDistanceMatrix()
				mol.coul_matrix = mol.properties["charges"] * np.expand_dims(mol.properties["charges"], axis=-1)
				for j in range(mol.NAtoms()):
					mol.DistMatrix[j,j] = 1.0
					mol.coul_matrix[j,j] = 0.0
				dist = mol.DistMatrix * 1.889725989
				dist = np.where(np.less(dist, srange_inner), np.ones_like(dist) * srange_inner, dist)
				dist = np.where(np.greater(dist, lrange_outer), np.ones_like(dist) * lrange_outer, dist)
				dist2 = dist * dist
				dist3 = dist2 * dist
				dist4 = dist3 * dist
				dist5 = dist4 * dist
				dist6 = dist5 * dist
				dist7 = dist6 * dist
				kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
				mrange_energy = np.sum(kern * mol.coul_matrix, axis=1)
				lrange_energy = np.sum(mol.coul_matrix, axis=1) / lrange_outer
				coulomb_energy = np.sum((mrange_energy - lrange_energy) / 2.0)
				regres_energies[i] = mol.properties["energy"] - coulomb_energy
			else:
				regres_energies[i] = mol.properties["energy"]
		noa = np.sum(stoichs, axis=0)
		x,r = np.linalg.lstsq(stoichs,regres_energies)[:2]
		for element in elements:
			avg_energy[element] = x[element]

		self.energy_fit = np.zeros((self.mol_set.max_atomic_num()+1))
		for element in avg_energy.keys():
			self.energy_fit[element] = avg_energy[element]
		energies = self.energy_data - np.sum(self.energy_fit[self.Z_data], axis=1)
		self.energy_mean = np.mean(energies)
		self.energy_std = np.std(energies)
		print("---- Results of Stoichiometric Model ----")
		print("MeanE  Energy: ", np.mean(energies))
		print("StdE  Energy: ", np.std(energies))
		print("MXE  Energy: ", np.max(energies))
		print("MNE  Energy: ", np.min(energies))
		print("RMSE Energy: ", np.sqrt(np.average(np.square(energies))))
		print("AvE: ", avg_energy)

		if self.train_charges:
			self.charge_mean = np.zeros((self.mol_set.max_atomic_num()+1))
			self.charge_std = np.zeros((self.mol_set.max_atomic_num()+1))
			for element in self.elements:
				element_idxs = np.where(np.equal(self.Z_data, element))
				element_charges = self.charges_data[element_idxs]
				self.charge_mean[element] = np.mean(element_charges)
				self.charge_std[element] = np.std(element_charges)
		self.embed_shape = self.codes_shape * (self.radial_rs.shape[0] + self.angular_rs.shape[0] * self.theta_s.shape[0])
		self.label_shape = 1
		return

	def calculate_coulomb_energy(self, dxyzs, q1q2, scatter_idx):
		"""
		Polynomial cutoff 1/r (in BOHR) obeying:
		kern = 1/r at SROuter and LRInner
		d(kern) = d(1/r) (true force) at SROuter,LRInner
		d**2(kern) = d**2(1/r) at SROuter and LRInner.
		d(kern) = 0 (no force) at/beyond SRInner and LROuter

		The hard cutoff is LROuter
		"""
		srange_inner = tf.constant(4.0*1.889725989, dtype=self.tf_precision)
		srange_outer = tf.constant(6.5*1.889725989, dtype=self.tf_precision)
		lrange_inner = tf.constant(13.0*1.889725989, dtype=self.tf_precision)
		lrange_outer = tf.constant(15.0*1.889725989, dtype=self.tf_precision)
		a, b, c, d, e, f, g, h = -9.83315, 4.49307, -0.784438, 0.0747019, -0.00419095, 0.000138593, -2.50374e-6, 1.90818e-8
		dist = tf.norm(dxyzs+1.e-16, axis=-1)
		dist *= 1.889725989
		dist = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) * srange_inner, dist)
		dist = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) * lrange_outer, dist)
		dist2 = dist * dist
		dist3 = dist2 * dist
		dist4 = dist3 * dist
		dist5 = dist4 * dist
		dist6 = dist5 * dist
		dist7 = dist6 * dist
		kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
		mrange_energy = tf.reduce_sum(kern * q1q2, axis=1)
		lrange_energy = tf.reduce_sum(q1q2, axis=1) / lrange_outer
		coulomb_energy = (mrange_energy - lrange_energy) / 2.0
		return tf.reduce_sum(tf.scatter_nd(scatter_idx, coulomb_energy, [self.batch_size, self.max_num_atoms]), axis=-1)

	def evaluate_set(self, mset):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mset.MaxNAtom()
			self.batch_size = 100
			self.eval_prepare()
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			energy_data[i] = mol.properties["energy"]
			gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0
		energy_true, energy_pred = [], []
		gradients_true, gradient_preds = [], []
		charges_true, charge_preds = [], []
		for ministep in range(int(num_mols / self.batch_size)):
			eval_pointer += self.batch_size
			batch_xyzs = xyz_data[eval_pointer - self.batch_size:eval_pointer]
			batch_Zs = Z_data[eval_pointer - self.batch_size:eval_pointer]
			nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
			batch_data = []
			batch_data.append(batch_xyzs)
			batch_data.append(batch_Zs)
			batch_data.append(nn_pairs)
			batch_data.append(nn_triples)
			batch_data.append(coulomb_pairs)
			batch_data.append(num_atoms_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(energy_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(gradient_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(charges_data[eval_pointer - self.batch_size:eval_pointer])
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_charges:
				total_energy, energy_label, gradients, gradient_labels, charges, charge_labels = self.sess.run([self.total_energy,
					self.energy_pl, self.gradients, self.gradient_labels, self.charges, self.charge_labels], feed_dict=feed_dict)
				charges_true.append(charge_labels)
				charge_preds.append(charges)
			else:
				total_energy, energy_label, gradients, gradient_labels = self.sess.run([self.total_energy,
					self.energy_pl, self.gradients, self.gradient_labels], feed_dict=feed_dict)
			energy_true.append(energy_label)
			energy_pred.append(total_energy)
			gradients_true.append(gradient_labels)
			gradient_preds.append(gradients)
		energy_true = np.concatenate(energy_true)
		energy_pred = np.concatenate(energy_pred)
		gradients_true = np.concatenate(gradients_true)
		gradient_preds = np.concatenate(gradient_preds)
		energy_errors = energy_true - energy_pred
		gradient_errors = gradients_true - gradient_preds
		if self.train_charges:
			charges_true = np.concatenate(charges_true)
			charge_preds = np.concatenate(charge_preds)
			charge_errors = charges_true - charge_preds
			return energy_errors, gradient_errors, charge_errors
		else:
			return energy_errors, gradient_errors

	def get_atom_energies(self, mset):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mset.MaxNAtom()
			self.batch_size = 100
			self.eval_prepare()
		atom_energy_data = []
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			energy_data[i] = mol.properties["energy"]
			gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0
		# energy_true, energy_pred = [], []
		# gradients_true, gradient_preds = [], []
		# charges_true, charge_preds = [], []
		for ministep in range(int(num_mols / self.batch_size)):
			eval_pointer += self.batch_size
			batch_xyzs = xyz_data[eval_pointer - self.batch_size:eval_pointer]
			batch_Zs = Z_data[eval_pointer - self.batch_size:eval_pointer]
			nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
			batch_data = []
			batch_data.append(batch_xyzs)
			batch_data.append(batch_Zs)
			batch_data.append(nn_pairs)
			batch_data.append(nn_triples)
			batch_data.append(coulomb_pairs)
			batch_data.append(num_atoms_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(energy_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(gradient_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(charges_data[eval_pointer - self.batch_size:eval_pointer])
			feed_dict = self.fill_feed_dict(batch_data)
			batch_atom_energies = self.sess.run(self.atom_nn_energy_tmp,
				feed_dict=feed_dict)
			atom_energy_data.append(batch_atom_energies)
		atom_energy_data = np.concatenate(atom_energy_data)
		Z_data_tmp = Z_data[:np.shape(atom_energy_data)[0]]
		elements = mset.AtomTypes().tolist()
		element_energy_data = []
		for element in elements:
			element_idxs = np.where(np.equal(Z_data_tmp, element))
			atom_energies = atom_energy_data[element_idxs]
			element_energy_data.append(atom_energies)
		return element_energy_data

	def get_energy_force_function(self,mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.eval_prepare()
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		num_atoms_data[0] = mol.NAtoms()
		Z_data[0][:mol.NAtoms()] = mol.atoms
		def EF(xyz_, DoForce=True):
			xyz_data[0][:mol.NAtoms()] = xyz_
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 15.0, self.max_num_atoms, True, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs, self.num_atoms_pl:num_atoms_data}
			if (DoForce):
				energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
				return energy[0], -JOULEPERHARTREE*gradients
			else:
				energy = self.sess.run(self.total_energy, feed_dict=feed_dict)
				return energy[0]
		return EF

	def evaluate_alchem_mol(self, mols):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = max([mol.NAtoms() for mol in mols])
			self.batch_size = 1
			self.alchem_prepare()
		xyz_data = np.zeros((len(mols), self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((len(mols), self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((len(mols)), dtype = np.int32)
		max_alchem_atoms = np.argmax(num_atoms_data)
		def alchem_energy_force(mols, delta, return_forces=True):
			for i, mol in enumerate(mols):
				xyz_data[i][:mol.NAtoms()] = mols[i].coords
				Z_data[i][:mol.NAtoms()] = mols[i].atoms
				num_atoms_data[i] = mol.NAtoms()
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(xyz_data[max_alchem_atoms:max_alchem_atoms+1],
							Z_data[max_alchem_atoms:max_alchem_atoms+1], 15.0, self.max_num_atoms, True, False)
			delta = np.array(delta).reshape(1)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs,
						self.num_atoms_pl:num_atoms_data, self.delta_pl:delta}
			energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
			return energy[0], -gradients
		return alchem_energy_force

	def element_opt(self, mol, replace_idx):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.element_opt_prepare()
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		Z_data[0][:mol.NAtoms()] = mol.atoms
		num_atoms_data[0] = mol.NAtoms()
		replace_idx_data = np.array(replace_idx, dtype=np.int32)
		xyz_data[0][:mol.NAtoms()] = mol.coords
		nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 15.0, self.max_num_atoms, True, False)
		total_loss = 10.0
		feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
					self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs,
					self.num_atoms_pl:num_atoms_data, self.replace_idx_pl:replace_idx_data}
		while total_loss > 0.0005:
			(_, total_loss, scalars_loss, lagrange1_loss, lagrange2_loss, zero_loss, energy,
				replace_scalars, lagrange_mult1, lagrange_mult2, zero_bound) = self.sess.run([self.minimize_op, self.total_loss,
				self.scalars_loss, self.lagrange1_loss, self.lagrange2_loss, self.zero_loss,
				self.total_energy, self.replace_scalars, self.lagrange_mult1, self.lagrange_mult2,
				self.zero_bound], feed_dict=feed_dict)
			LOGGER.info("Total loss: %11.8f  Scalars loss: %11.8f  Lagrange1 loss: %11.8f  Lagrange2 loss: %11.8f  Zero loss: %11.8f",
				total_loss, scalars_loss, lagrange1_loss, lagrange2_loss, zero_loss)
			LOGGER.info("Atomization Energy: %11.8f  Lagrange Multiplier: %11.8f  Sum of Scalars: %11.8f",
				energy, lagrange_mult1, np.sum(replace_scalars))
		print(replace_scalars)
		return

class UniversalNetwork_v3(UniversalNetwork_v2):
	"""
	The 0.2 model chemistry.
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
		self.codes_shape = self.element_codes.shape[1]
		self.assign_activation()

		#Reloads a previous network if name variable is not None
		if name != None:
			self.name = name
			self.load_network()
			self.path = PARAMS["networks_directory"]
			self.network_directory = PARAMS["networks_directory"]+self.name
			LOGGER.info("Reloaded network from %s", self.network_directory)
			return

		#Data parameters
		self.mol_set_name = mol_set_name
		self.mol_set = MSet(self.mol_set_name)
		self.mol_set.Load()
		self.elements = self.mol_set.AtomTypes()
		self.max_num_atoms = self.mol_set.MaxNAtom()
		self.num_molecules = len(self.mol_set.mols)
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

	def compute_normalization(self):
		elements = self.mol_set.AtomTypes().tolist()
		self.energy_fit = np.zeros((self.mol_set.max_atomic_num()+1))
		for element in ele_U_wb97xd.keys():
			self.energy_fit[element] = ele_U_wb97xd[element]
		energies = self.energy_data - np.sum(self.energy_fit[self.Z_data], axis=1)
		self.energy_mean = np.mean(energies)
		self.energy_std = np.std(energies)
		print("---- Results of Stoichiometric Model ----")
		print("MeanE  Energy: ", np.mean(energies))
		print("StdE  Energy: ", np.std(energies))
		print("MXE  Energy: ", np.max(energies))
		print("MNE  Energy: ", np.min(energies))
		print("RMSE Energy: ", np.sqrt(np.average(np.square(energies))))

		if self.train_charges:
			self.charge_mean = np.zeros((self.mol_set.max_atomic_num()+1))
			self.charge_std = np.zeros((self.mol_set.max_atomic_num()+1))
			for element in self.elements:
				element_idxs = np.where(np.equal(self.Z_data, element))
				element_charges = self.charges_data[element_idxs]
				self.charge_mean[element] = np.mean(element_charges)
				self.charge_std[element] = np.std(element_charges)
		self.embed_shape = self.codes_shape * (self.radial_rs.shape[0] + self.angular_rs.shape[0] * self.theta_s.shape[0])
		self.label_shape = 1
		return

	def train(self):
		if self.train_charges:
			for i in range(250):
				self.step += 1
				self.charge_train_step(self.step)
				if self.step%self.test_freq==0:
					test_loss = self.charge_test_step(self.step)
					if self.step == self.test_freq:
						self.best_loss = test_loss
						self.save_checkpoint(self.step)
					elif test_loss < self.best_loss:
						self.best_loss = test_loss
						self.save_checkpoint(self.step)
		self.step = 0
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

			self.elements = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
			num_elements = self.elements.get_shape().as_list()[0]
			self.element_codes = tf.Variable(self.element_codes, trainable=False, dtype=self.tf_precision, name="element_codes")
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.get_variable("energy_mean", shape=self.mol_set.max_atomic_num()+1, dtype=self.tf_precision,
				initializer=tf.zeros_initializer())
			energy_std = tf.get_variable("energy_std", shape=self.mol_set.max_atomic_num()+1, dtype=self.tf_precision,
				initializer=tf.ones_initializer())
			#energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			#energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
					self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
					self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
					self.charge_loss = self.loss_op(self.charges - self.charge_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
					# tf.summary.scalar("charge_loss", self.charge_loss)
					# tf.add_to_collection('total_loss', self.charge_loss)
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
					self.charge_train_op = self.optimizer(self.charge_loss, self.learning_rate, self.momentum, charge_variables)
			with tf.name_scope('energy_inference'):
				atom_nn_energy, energy_variables = self.energy_inference(embed, atom_codes, padding_mask)
				atom_energy_mean, atom_energy_std = tf.gather(energy_mean, self.Zs_pl), tf.gather(energy_std, self.Zs_pl)
				self.atom_nn_energy = (atom_nn_energy * tf.square(atom_energy_std)) + atom_energy_mean
				self.mol_nn_energy = tf.reduce_sum(self.atom_nn_energy, axis=1)
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
				energy_variables.append(energy_mean)
				energy_variables.append(energy_std)
			if self.train_charges:
				dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
				self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
				self.total_energy += self.mol_coulomb_energy
			self.energy_loss = 100.0 * self.loss_op(self.total_energy - self.energy_pl) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
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

			self.train_op = self.optimizer(self.total_loss, self.learning_rate, self.momentum, energy_variables)
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

	def eval_prepare(self):
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

			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)

			self.elements = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
			num_elements = self.elements.get_shape().as_list()[0]
			self.element_codes = tf.Variable(self.element_codes, trainable=False, dtype=self.tf_precision, name="element_codes")
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.get_variable("energy_mean", shape=self.mol_set.max_atomic_num()+1, dtype=self.tf_precision,
										  initializer=tf.zeros_initializer())
			energy_std = tf.get_variable("energy_std", shape=self.mol_set.max_atomic_num()+1, dtype=self.tf_precision,
										 initializer=tf.ones_initializer())
			#energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			#energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
											  self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
			with tf.name_scope('energy_inference'):
				atom_nn_energy, energy_variables = self.energy_inference(embed, atom_codes, padding_mask)
				atom_energy_mean, atom_energy_std = tf.gather(energy_mean, self.Zs_pl), tf.gather(energy_std, self.Zs_pl)
				self.atom_nn_energy = (atom_nn_energy * tf.square(atom_energy_std)) + atom_energy_mean
				self.mol_nn_energy = tf.reduce_sum(self.atom_nn_energy, axis=1)
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
				self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
				self.total_energy += self.mol_coulomb_energy
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)

			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

#	def eval_prepare(self):
#		"""
#		Get placeholders, graph and losses in order to begin training.
#		Also assigns the desired padding.
#
#		Args:
#			continue_training: should read the graph variables from a saved checkpoint.
#		"""
#		with tf.Graph().as_default():
#			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
#			self.Zs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms])
#			self.nn_pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None])
#			self.nn_triples_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None, 2])
#			self.coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None])
#			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
#			self.energy_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size])
#			self.gradients_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
#			self.charges_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms])
#
#			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
#			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
#			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
#			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
#			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
#			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
#			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
#
#			self.elements = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
#			num_elements = self.elements.get_shape().as_list()[0]
#			self.element_codes = tf.Variable(self.element_codes, trainable=False, dtype=self.tf_precision, name="element_codes")
#			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
#			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
#			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
#			if self.train_charges:
#				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
#				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)
#
#			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
#			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
#					self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
#			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
#			if self.train_charges:
#				with tf.name_scope('charge_inference'):
#					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
#					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
#					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
#					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
#					self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
#					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
#			with tf.name_scope('energy_inference'):
#				self.atom_nn_energy, energy_variables = self.energy_inference(embed, atom_codes, padding_mask)
#				self.mol_nn_energy = tf.reduce_sum(self.atom_nn_energy, axis=1)
#				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
#				self.mol_nn_energy += mol_energy_fit
#				self.total_energy = self.mol_nn_energy
#			if self.train_charges:
#				dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
#				self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
#				self.total_energy += self.mol_coulomb_energy
#			with tf.name_scope('gradients'):
#				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
#				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
#				self.gradient_labels = tf.gather_nd(self.gradients_pl, padding_mask)
#			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
#			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
#		return

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
		activation_list=[]
		with tf.variable_scope("energy_network", reuse=tf.AUTO_REUSE):
			code_kernel1 = tf.get_variable(name="CodeKernel1", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
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
						activation_list.append(activations)
						variables.append(weights)
						variables.append(biases)
				else:
					with tf.name_scope('hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(activations, weights) + biases)
						activation_list.append(activations)
						variables.append(weights)
						variables.append(biases)
			with tf.name_scope('regression_linear'):
				activations = tf.concat(activation_list, axis=-1)
				weights = self.variable_with_weight_decay(shape=[sum(self.hidden_layers), 1],
						stddev=math.sqrt(2.0 / float(sum(self.hidden_layers))), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
				outputs = tf.squeeze(tf.matmul(activations, weights) + biases, axis=1)
				variables.append(weights)
				variables.append(biases)
				atom_nn_energy = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
		return atom_nn_energy, variables

#	def energy_inference(self, embed, atom_codes, indices):
#		"""
#		Builds a Behler-Parinello graph
#
#		Args:
#			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
#			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
#		Returns:
#			The BP graph output
#		"""
#		variables=[]
#		with tf.variable_scope("energy_network", reuse=tf.AUTO_REUSE):
#			#code_kernel1 = tf.get_variable(name="CodeKernel1", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
#			#code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
#			#variables.append(code_kernel1)
#			#variables.append(code_kernel2)
#			#coded_weights = tf.matmul(atom_codes, code_kernel1)
#			#coded_embed = tf.einsum('ijk,ij->ijk', embed, coded_weights)
#			#coded_embed = tf.reshape(tf.einsum('ijk,jl->ilk', coded_embed, code_kernel2), [tf.shape(embed)[0], -1])
#			embed = tf.reshape(embed, [tf.shape(embed)[0], -1])
#			for i in range(len(self.hidden_layers)):
#				with tf.name_scope('hidden'+str(i+1)):
#					bias_init = tf.get_variable(name='bias_init'+str(i), shape=(self.codes_shape, self.hidden_layers[i]), dtype=self.tf_precision,
#						initializer=tf.zeros_initializer())
#					biases = self.activation_function(tf.matmul(atom_codes, bias_init))
#					if i == 0:
#						weights = tf.get_variable(name='weights'+str(i), shape=(self.embed_shape, self.hidden_layers[i]),
#							dtype=self.tf_precision)
#						#weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[i]],
#						#		stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
#						activations = self.activation_function(tf.matmul(embed, weights) + biases)
#					else:
#						weights = tf.get_variable(name='weights', shape=(self.hidden_layers[i-1], self.hidden_layers[i]),
#							dtype=self.tf_precision)
#						#weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
#						#		stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
#						activations = self.activation_function(tf.matmul(activations, weights) + biases)
#					variables.append(weights)
#					variables.append(bias_init)
#			with tf.name_scope('regression_linear'):
#				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
#						stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
#				biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
#				outputs = tf.squeeze(tf.matmul(activations, weights) + biases, axis=1)
#				variables.append(weights)
#				variables.append(biases)
#				atom_nn_energy = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
#		return atom_nn_energy, variables

	def charge_inference(self, embed, atom_codes, indices):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		variables=[]
		with tf.variable_scope("charge_network", reuse=tf.AUTO_REUSE):
			#code_kernel1 = tf.get_variable(name="CodeKernel", shape=(self.codes_shape, self.codes_shape),dtype=self.tf_precision)
			#code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(self.codes_shape, self.codes_shape),dtype=self.tf_precision)
			#variables.append(code_kernel1)
			#variables.append(code_kernel2)
			#coded_weights = tf.matmul(atom_codes, code_kernel1)
			#coded_embed = tf.einsum('ijk,ij->ijk', embed, coded_weights)
			#coded_embed = tf.reshape(tf.einsum('ijk,jl->ilk', coded_embed, code_kernel2), [tf.shape(embed)[0], -1])
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
				atom_nn_charges = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
		return atom_nn_charges, variables

	def calculate_coulomb_energy(self, dxyzs, q1q2, scatter_idx):
		"""
		Polynomial cutoff 1/r (in BOHR) obeying:
		kern = 1/r at SROuter and LRInner
		d(kern) = d(1/r) (true force) at SROuter,LRInner
		d**2(kern) = d**2(1/r) at SROuter and LRInner.
		d(kern) = 0 (no force) at/beyond SRInner and LROuter

		The hard cutoff is LROuter
		"""
		srange_inner = tf.constant(5.0*1.889725989, dtype=self.tf_precision)
		srange_outer = tf.constant(8.5*1.889725989, dtype=self.tf_precision)
		lrange_inner = tf.constant(17.0*1.889725989, dtype=self.tf_precision)
		lrange_outer = tf.constant(20.0*1.889725989, dtype=self.tf_precision)
		a, b, c, d, e, f, g, h = -7.59607, 2.70373, -0.357654, 0.0257835, -0.0010942, 0.000027356, -3.73478e-7, 2.15073e-9
		dist = tf.norm(dxyzs+1.e-16, axis=-1)
		dist *= 1.889725989
		dist = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) * srange_inner, dist)
		dist = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) * lrange_outer, dist)
		dist2 = dist * dist
		dist3 = dist2 * dist
		dist4 = dist3 * dist
		dist5 = dist4 * dist
		dist6 = dist5 * dist
		dist7 = dist6 * dist
		kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
		mrange_energy = tf.reduce_sum(kern * q1q2, axis=1)
		lrange_energy = tf.reduce_sum(q1q2, axis=1) / lrange_outer
		coulomb_energy = (mrange_energy - lrange_energy) / 2.0
		return tf.reduce_sum(tf.scatter_nd(scatter_idx, coulomb_energy, [self.batch_size, self.max_num_atoms]), axis=-1)

	def charge_train_step(self, step):
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
		for ministep in range (0, int(0.1 * Ncase_train/self.batch_size)):
			batch_data = self.get_train_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			_, charge_loss = self.sess.run([self.charge_train_op, self.charge_loss], feed_dict=feed_dict)
			train_charge_loss += charge_loss
			num_batches += 1
		train_charge_loss /= num_batches
		duration = time.time() - start_time
		self.print_charge_epoch(step, duration, train_charge_loss)
		return

	def charge_test_step(self, step):
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
		test_charge_loss = 0.0
		test_charge_labels, test_charge_outputs = [], []
		num_atoms_epoch = 0.0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.get_test_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			charges, charge_labels, charge_loss, num_atoms = self.sess.run([self.charges, self.charge_labels,
			self.charge_loss, self.num_atoms_pl],  feed_dict=feed_dict)
			test_charge_loss += charge_loss
			test_charge_labels.append(charge_labels)
			test_charge_outputs.append(charges)
			num_atoms_epoch += np.sum(num_atoms)
			num_batches += 1
		test_charge_loss /= num_batches
		duration = time.time() - start_time
		test_charge_labels = np.concatenate(test_charge_labels)
		test_charge_outputs = np.concatenate(test_charge_outputs)
		test_charge_errors = test_charge_labels - test_charge_outputs
		for i in [random.randint(0, num_atoms_epoch - 1) for _ in range(10)]:
			LOGGER.info("Charge label: %11.8f  Charge output: %11.8f", test_charge_labels[i], test_charge_outputs[i])
		LOGGER.info("MAE  Charges %11.8f", np.mean(np.abs(test_charge_errors)))
		LOGGER.info("MSE  Charges %11.8f", np.mean(test_charge_errors))
		LOGGER.info("RMSE Charges %11.8f", np.sqrt(np.mean(np.square(test_charge_errors))))
		self.print_charge_epoch(step, duration, test_charge_loss, testing=True)
		return test_charge_loss

	def print_charge_epoch(self, step, duration, charge_loss, testing=False):
		if testing:
			LOGGER.info("step: %5d  duration: %.3f  charge loss: %.10f",
			step, duration, charge_loss)
		else:
			LOGGER.info("step: %5d  duration: %.3f  charge loss: %.10f",
			step, duration, charge_loss)
		return

	def evaluate_set(self, mset):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mset.MaxNAtom()
			self.batch_size = 400
			self.reload_set()
			self.eval_prepare()
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			energy_data[i] = mol.properties["energy"]
			gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0
		energy_true, energy_pred = [], []
		gradients_true, gradient_preds = [], []
		charges_true, charge_preds = [], []
		for ministep in range(int(num_mols / self.batch_size)):
			eval_pointer += self.batch_size
			batch_xyzs = xyz_data[eval_pointer - self.batch_size:eval_pointer]
			batch_Zs = Z_data[eval_pointer - self.batch_size:eval_pointer]
			nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
			batch_data = []
			batch_data.append(batch_xyzs)
			batch_data.append(batch_Zs)
			batch_data.append(nn_pairs)
			batch_data.append(nn_triples)
			batch_data.append(coulomb_pairs)
			batch_data.append(num_atoms_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(energy_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(gradient_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(charges_data[eval_pointer - self.batch_size:eval_pointer])
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_charges:
				total_energy, energy_label, gradients, gradient_labels, charges, charge_labels = self.sess.run([self.total_energy,
					self.energy_pl, self.gradients, self.gradient_labels, self.charges, self.charge_labels], feed_dict=feed_dict)
				charges_true.append(charge_labels)
				charge_preds.append(charges)
			else:
				total_energy, energy_label, gradients, gradient_labels = self.sess.run([self.total_energy,
					self.energy_pl, self.gradients, self.gradient_labels], feed_dict=feed_dict)
			energy_true.append(energy_label)
			energy_pred.append(total_energy)
			gradients_true.append(gradient_labels)
			gradient_preds.append(gradients)
		energy_true = np.concatenate(energy_true)
		energy_pred = np.concatenate(energy_pred)
		gradients_true = np.concatenate(gradients_true)
		gradient_preds = np.concatenate(gradient_preds)
		energy_errors = energy_true - energy_pred
		gradient_errors = gradients_true - gradient_preds
		#for i, mol in enumerate(mset.mols):
		#	mol.properties["energy_error"] = energy_errors[i]
		if self.train_charges:
			charges_true = np.concatenate(charges_true)
			charge_preds = np.concatenate(charge_preds)
			charge_errors = charges_true - charge_preds
			return energy_errors, gradient_errors, charge_errors
		else:
			return energy_errors, gradient_errors

	def get_energy_force_function(self,mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.eval_prepare()
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		num_atoms_data[0] = mol.NAtoms()
		Z_data[0][:mol.NAtoms()] = mol.atoms
		def EF(xyz_, DoForce=True):
			xyz_data[0][:mol.NAtoms()] = xyz_
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 15.0, self.max_num_atoms, True, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs, self.num_atoms_pl:num_atoms_data}
			if (DoForce):
				energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
				return energy[0], -JOULEPERHARTREE*gradients
			else:
				energy = self.sess.run(self.total_energy, feed_dict=feed_dict)
				return energy[0]
		return EF


class UniversalNetwork_v4(UniversalNetwork):
	"""
	The 0.2 model chemistry.
	"""
	def __init__(self, mol_set_name=None, max_num_atoms=None, name=None):
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
		self.validation_ratio = PARAMS["validation_ratio"]
		self.test_ratio = PARAMS["TestRatio"]
		self.element_codes = ELEMENT_CODES4
		self.codes_shape = self.element_codes.shape[1]
		self.assign_activation()

		#Reloads a previous network if name variable is not None
		if name != None:
			self.name = name
			self.load_network()
			self.path = PARAMS["networks_directory"]
			self.network_directory = PARAMS["networks_directory"]+self.name
			#self.max_num_atoms = max_num_atoms if max_num_atoms else self.mol_set.MaxNAtom()
			LOGGER.info("Reloaded network from %s", self.network_directory)
			return

		#Data parameters
		self.mol_set_name = mol_set_name
		self.mol_set = MSet(self.mol_set_name)
		self.mol_set.Load()
		self.elements = self.mol_set.AtomTypes()
		self.max_num_atoms = max_num_atoms if max_num_atoms else self.mol_set.MaxNAtom()
		self.num_molecules = len(self.mol_set.mols)
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

	def train(self):
		for i in range(self.max_steps):
			self.step += 1
			self.train_step(self.step)
			if self.step%self.test_freq==0:
				validation_loss = self.validation_step(self.step)
				if self.step == self.test_freq:
					self.best_loss = validation_loss
					self.save_checkpoint(self.step)
					self.test_step(self.step)
				elif validation_loss < self.best_loss:
					self.best_loss = validation_loss
					self.save_checkpoint(self.step)
					self.test_step(self.step)
		self.sess.close()
		return

	def load_data(self):
		if (self.mol_set == None):
			try:
				self.reload_set()
			except Exception as Ex:
				print("TensorData object has no molecule set.", Ex)
		self.xyz_data = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype = np.float64)
		self.Z_data = np.zeros((self.num_molecules, self.max_num_atoms), dtype = np.int32)
		if self.train_charges:
			self.charges_data = np.zeros((self.num_molecules, self.max_num_atoms), dtype = np.float64)
		self.num_atoms_data = np.zeros((self.num_molecules), dtype = np.int32)
		self.energy_data = np.zeros((self.num_molecules), dtype = np.float64)
		self.gradient_data = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(self.mol_set.mols):
			self.xyz_data[i][:mol.NAtoms()] = mol.coords
			self.Z_data[i][:mol.NAtoms()] = mol.atoms
			self.energy_data[i] = mol.properties["energy"]
			self.gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			self.num_atoms_data[i] = mol.NAtoms()
			if self.train_charges:
				self.charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
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
		self.num_validation_cases = int(self.validation_ratio * self.num_molecules)
		self.num_test_cases = int(self.test_ratio * self.num_molecules)
		num_validation_test = self.num_validation_cases + self.num_test_cases
		self.num_train_cases = int(self.num_molecules - num_validation_test)
		case_idxs = np.arange(int(self.num_molecules))
		np.random.shuffle(case_idxs)
		self.validation_idxs = case_idxs[int(self.num_molecules - self.num_validation_cases):]
		self.test_idxs = case_idxs[int(self.num_molecules - num_validation_test):int(self.num_molecules - self.num_validation_cases)]
		self.train_idxs = case_idxs[:int(self.num_molecules - num_validation_test)]
		self.train_pointer, self.test_pointer, self.validation_pointer = 0, 0, 0
		if self.batch_size > self.num_train_cases:
			raise Exception("Insufficent training data to fill a training batch.\n"\
					+str(self.num_train_cases)+" cases in dataset with a batch size of "+str(self.batch_size))
		if self.batch_size > self.num_validation_cases:
			raise Exception("Insufficent testing data to fill a validation batch.\n"\
					+str(self.num_validation_cases)+" cases in dataset with a batch size of "+str(self.batch_size))
		LOGGER.debug("Number of training cases: %i", self.num_train_cases)
		LOGGER.debug("Number of validation cases: %i", self.num_validation_cases)
		LOGGER.debug("Number of test cases: %i", self.num_test_cases)
		return

	def get_train_batch(self):
		if self.train_pointer + self.batch_size >= self.num_train_cases:
			np.random.shuffle(self.train_idxs)
			self.train_pointer = 0
		self.train_pointer += self.batch_size
		batch_xyzs = self.xyz_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]]
		batch_Zs = self.Z_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]]
		nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
		batch_data = []
		batch_data.append(batch_xyzs)
		batch_data.append(batch_Zs)
		batch_data.append(nn_pairs)
		batch_data.append(nn_triples)
		batch_data.append(coulomb_pairs)
		batch_data.append(self.num_atoms_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		batch_data.append(self.energy_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		batch_data.append(self.gradient_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		if self.train_charges:
			batch_data.append(self.charges_data[self.train_idxs[self.train_pointer - self.batch_size:self.train_pointer]])
		return batch_data

	def get_validation_batch(self):
		if self.validation_pointer + self.batch_size >= self.num_validation_cases:
			self.validation_pointer = 0
		self.validation_pointer += self.batch_size
		batch_xyzs = self.xyz_data[self.validation_idxs[self.validation_pointer - self.batch_size:self.validation_pointer]]
		batch_Zs = self.Z_data[self.validation_idxs[self.validation_pointer - self.batch_size:self.validation_pointer]]
		nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
		batch_data = []
		batch_data.append(batch_xyzs)
		batch_data.append(batch_Zs)
		batch_data.append(nn_pairs)
		batch_data.append(nn_triples)
		batch_data.append(coulomb_pairs)
		batch_data.append(self.num_atoms_data[self.validation_idxs[self.validation_pointer - self.batch_size:self.validation_pointer]])
		batch_data.append(self.energy_data[self.validation_idxs[self.validation_pointer - self.batch_size:self.validation_pointer]])
		batch_data.append(self.gradient_data[self.validation_idxs[self.validation_pointer - self.batch_size:self.validation_pointer]])
		if self.train_charges:
			batch_data.append(self.charges_data[self.validation_idxs[self.validation_pointer - self.batch_size:self.validation_pointer]])
		return batch_data

	def get_test_batch(self):
		if self.test_pointer + self.batch_size >= self.num_test_cases:
			self.test_pointer = 0
		self.test_pointer += self.batch_size
		batch_xyzs = self.xyz_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]]
		batch_Zs = self.Z_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]]
		nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
		batch_data = []
		batch_data.append(batch_xyzs)
		batch_data.append(batch_Zs)
		batch_data.append(nn_pairs)
		batch_data.append(nn_triples)
		batch_data.append(coulomb_pairs)
		batch_data.append(self.num_atoms_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]])
		batch_data.append(self.energy_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]])
		batch_data.append(self.gradient_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]])
		if self.train_charges:
			batch_data.append(self.charges_data[self.test_idxs[self.test_pointer - self.batch_size:self.test_pointer]])
		return batch_data

	def validation_step(self, step):
		"""
		Perform a single validation step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		validation_loss =  0.0
		start_time = time.time()
		num_batches = 0
		validation_energy_loss = 0.0
		validation_gradient_loss = 0.0
		validation_charge_loss = 0.0
		validation_energy_labels, validation_energy_outputs = [], []
		validation_force_labels, validation_force_outputs = [], []
		validation_charge_labels, validation_charge_outputs = [], []
		num_atoms_epoch = 0.0
		for ministep in range (0, int(self.num_validation_cases/self.batch_size)):
			batch_data = self.get_validation_batch()
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_charges:
				total_energies, energy_labels, gradients, gradient_labels, charges, charge_labels, total_loss, energy_loss, gradient_loss, charge_loss, num_atoms = self.sess.run([self.total_energy,
				self.energy_pl, self.gradients, self.gradient_labels, self.charges, self.charge_labels,
				self.total_loss, self.energy_loss, self.gradient_loss, self.charge_loss, self.num_atoms_pl],  feed_dict=feed_dict)
				validation_charge_loss += charge_loss
				validation_charge_labels.append(charge_labels)
				validation_charge_outputs.append(charges)
			else:
				total_energies, energy_labels, gradients, gradient_labels, total_loss, energy_loss, gradient_loss, num_atoms = self.sess.run([self.total_energy,
				self.energy_pl, self.gradients, self.gradient_labels, self.total_loss, self.energy_loss,
				self.gradient_loss, self.num_atoms_pl],  feed_dict=feed_dict)
			validation_loss += total_loss
			validation_energy_loss += energy_loss
			validation_gradient_loss += gradient_loss
			validation_energy_labels.append(energy_labels)
			validation_energy_outputs.append(total_energies)
			validation_force_labels.append(-1.0 * gradient_labels)
			validation_force_outputs.append(-1.0 * gradients)
			num_atoms_epoch += np.sum(num_atoms)
			num_batches += 1
		validation_loss /= num_batches
		validation_energy_loss /= num_batches
		validation_gradient_loss /= num_batches
		validation_charge_loss /= num_batches
		validation_energy_labels = np.concatenate(validation_energy_labels)
		validation_energy_outputs = np.concatenate(validation_energy_outputs)
		validation_energy_errors = validation_energy_labels - validation_energy_outputs
		validation_force_labels = np.concatenate(validation_force_labels)
		validation_force_outputs = np.concatenate(validation_force_outputs)
		validation_force_errors = validation_force_labels - validation_force_outputs
		duration = time.time() - start_time
		for i in [random.randint(0, num_batches * self.batch_size - 1) for _ in range(10)]:
			LOGGER.info("Energy label: %12.8f  Energy output: %12.8f", validation_energy_labels[i], validation_energy_outputs[i])
		for i in [random.randint(0, num_atoms_epoch - 1) for _ in range(10)]:
			LOGGER.info("Forces label: %s  Forces output: %s", validation_force_labels[i], validation_force_outputs[i])
		if self.train_charges:
			validation_charge_labels = np.concatenate(validation_charge_labels)
			validation_charge_outputs = np.concatenate(validation_charge_outputs)
			validation_charge_errors = validation_charge_labels - validation_charge_outputs
			for i in [random.randint(0, num_atoms_epoch - 1) for _ in range(10)]:
				LOGGER.info("Charge label: %11.8f  Charge output: %11.8f", validation_charge_labels[i], validation_charge_outputs[i])
			LOGGER.info("MAE  Energy: %11.8f  Forces: %11.8f  Charges %11.8f", np.mean(np.abs(validation_energy_errors)),
			np.mean(np.abs(validation_force_errors)), np.mean(np.abs(validation_charge_errors)))
			LOGGER.info("MSE  Energy: %11.8f  Forces: %11.8f  Charges %11.8f", np.mean(validation_energy_errors),
			np.mean(validation_force_errors), np.mean(validation_charge_errors))
			LOGGER.info("RMSE Energy: %11.8f  Forces: %11.8f  Charges %11.8f", np.sqrt(np.mean(np.square(validation_energy_errors))),
			np.sqrt(np.mean(np.square(validation_force_errors))), np.sqrt(np.mean(np.square(validation_charge_errors))))
		else:
			LOGGER.info("MAE  Energy: %11.8f  Forces: %11.8f", np.mean(np.abs(validation_energy_errors)),
			np.mean(np.abs(validation_force_errors)))
			LOGGER.info("MSE  Energy: %11.8f  Forces: %11.8f", np.mean(validation_energy_errors),
			np.mean(validation_force_errors))
			LOGGER.info("RMSE Energy: %11.8f  Forces: %11.8f", np.sqrt(np.mean(np.square(validation_energy_errors))),
			np.sqrt(np.mean(np.square(validation_force_errors))))
		self.print_epoch(step, duration, validation_loss, validation_energy_loss, validation_gradient_loss, validation_charge_loss, testing=True)
		return validation_loss

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
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, atom_nn_charges, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std)
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
				atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
				self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
				self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
				dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
				self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
				self.total_energy += self.mol_coulomb_energy
				self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
				self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
				self.charge_loss = 0.1 * self.loss_op(self.charges - self.charge_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
				tf.summary.scalar("charge_loss", self.charge_loss)
				tf.add_to_collection('total_loss', self.charge_loss)
			#if self.train_charges:
			#	with tf.name_scope('charge_inference'):
			#		atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
			#		atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
			#		self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
			#		self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
			#		dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
			#		self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
			#		self.total_energy += self.mol_coulomb_energy
			#		self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
			#		self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
			#		self.charge_loss = 0.1 * self.loss_op(self.charges - self.charge_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
			#		tf.summary.scalar("charge_loss", self.charge_loss)
			#		tf.add_to_collection('total_loss', self.charge_loss)
			self.energy_loss = 100.0 * self.loss_op(self.total_energy - self.energy_pl) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
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

	def eval_prepare(self):
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
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.atom_nn_energy_tmp = self.atom_nn_energy * energy_std
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std) + energy_mean
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
					dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
					self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
				self.gradient_labels = tf.gather_nd(self.gradients_pl, padding_mask)

			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def alchem_prepare(self):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[None, self.max_num_atoms, 3])
			self.Zs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms])
			self.nn_pairs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None])
			self.nn_triples_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None, 2])
			self.coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[None])
			self.delta_pl = tf.placeholder(self.tf_precision, shape=[1])

			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=True, dtype=self.tf_precision, name="element_codes")
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			alchem_padding_mask = tf.where(tf.reduce_any(tf.not_equal(self.Zs_pl, 0), axis=0, keepdims=True))
			_, max_atom_idx = tf.nn.top_k(self.num_atoms_pl)
			self.alchem_xyzs = tf.gather(self.xyzs_pl, max_atom_idx)
			self.alchem_switch = tf.where(tf.not_equal(self.Zs_pl, 0), tf.stack([tf.tile(1.0 - self.delta_pl,
								[self.max_num_atoms]), tf.tile(self.delta_pl, [self.max_num_atoms])]),
								tf.zeros_like(self.Zs_pl, dtype=eval(PARAMS["tf_prec"])))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			reconst_embed = tf.scatter_nd(padding_mask, embed, [tf.cast(tf.shape(self.Zs_pl)[0], tf.int64),
				self.max_num_atoms, self.codes_shape, int(self.embed_shape / self.codes_shape)])
			alchem_embed = tf.reduce_sum(tf.stack([reconst_embed[0] * (1.0 - self.delta_pl),
				reconst_embed[1] * self.delta_pl], axis=0), axis=0)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			reconst_atom_codes = tf.scatter_nd(padding_mask, atom_codes,
				[tf.cast(tf.shape(self.Zs_pl)[0], tf.int64), self.max_num_atoms, self.codes_shape])
			alchem_atom_codes = tf.reduce_sum(tf.stack([reconst_atom_codes[0] * (1.0 - self.delta_pl),
				reconst_atom_codes[1] * self.delta_pl], axis=0), axis=0)
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(alchem_embed,
					alchem_atom_codes, alchem_padding_mask)
				self.atom_nn_energy *= tf.reduce_sum(self.alchem_switch, axis=0)
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std) + energy_mean
				mol_energy_fit = tf.reduce_sum(tf.reduce_sum(tf.gather(energy_fit,
					self.Zs_pl) * self.alchem_switch, axis=0), axis=0)
				# self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(alchem_embed,
						alchem_atom_codes, alchem_padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					atom_charge_mean *= self.alchem_switch
					atom_charge_std *= self.alchem_switch
					atom_charge_mean = tf.reduce_sum(atom_charge_mean, axis=0)
					atom_charge_std = tf.reduce_sum(atom_charge_std, axis=0)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					num_alchem_atoms = tf.reduce_sum(self.alchem_switch)
					self.atom_nn_charges = self.alchem_charge_equalization(self.atom_nn_charges,
						num_alchem_atoms, tf.reduce_sum(self.alchem_switch, axis=0))
					dxyzs, q1q2, scatter_coulomb = self.alchem_gather_coulomb(self.alchem_xyzs,
						self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
			with tf.name_scope('gradients'):
				self.gradients = tf.reduce_sum(tf.gradients(self.total_energy, self.xyzs_pl)[0], axis=0)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def element_opt_prepare(self):
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
			self.replace_idx_pl = tf.placeholder(tf.int32, shape=[1, 2])

			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=True,
				dtype=self.tf_precision, name="element_codes")
			elements = tf.constant(self.elements, dtype = tf.int32)
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_std = tf.Variable(self.energy_std, trainable=False, dtype = self.tf_precision)
			if self.train_charges:
				charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
				charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			self.replace_scalars = tf.Variable([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], trainable=True,
				dtype=self.tf_precision, name="replace_scalars")
			self.lagrange_mult1 = tf.Variable(1.0, trainable=True, dtype=self.tf_precision,
				name="lagrange_mult1")
			self.lagrange_mult2 = tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
				1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], trainable=True,
				dtype=self.tf_precision, name="lagrange_mult2")
			self.lagrange_mult3 = tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
				1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], trainable=True,
				dtype=self.tf_precision, name="lagrange_mult3")
			self.zero_bound = tf.Variable(self.replace_scalars, trainable=True,
				dtype=self.tf_precision, name="zero_bound")
			self.one_bound = tf.Variable(1.0 - self.replace_scalars, trainable=True,
				dtype=self.tf_precision, name="one_bound")

			self.replace_codes = tf.reduce_sum(tf.expand_dims(self.replace_scalars, axis=-1)
				* tf.gather(self.element_codes, elements), axis=0)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = tf_sym_func_element_codes_replace(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl,
					self.nn_triples_pl, self.element_codes, radial_gauss, radial_cutoff, angular_gauss,
					thetas, angular_cutoff, zeta, eta, self.replace_idx_pl, self.replace_codes)
			atom_codes = tf.gather(self.element_codes, self.Zs_pl)
			atom_idx = tf.range(self.num_atoms_pl[0])
			replace_broadcast = tf.where(tf.equal(atom_idx, self.replace_idx_pl[0,1]),
				tf.ones_like(atom_idx, dtype=self.tf_precision),
				tf.zeros_like(atom_idx, dtype=self.tf_precision))
			codes_broadcast = tf.where(tf.equal(atom_idx, self.replace_idx_pl[0,1]),
				tf.zeros_like(atom_idx, dtype=self.tf_precision),
				tf.ones_like(atom_idx, dtype=self.tf_precision))
			atom_codes = (tf.expand_dims(replace_broadcast, axis=-1) * self.replace_codes
				+ tf.expand_dims(codes_broadcast, axis=-1) * atom_codes)
			atom_codes = tf.gather_nd(atom_codes, padding_mask)
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std) + energy_mean
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				# self.mol_nn_energy += mol_energy_fit
				self.total_energy = self.mol_nn_energy
			if self.train_charges:
				with tf.name_scope('charge_inference'):
					atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
					atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
					self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
					self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
					dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
					self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
					self.total_energy += self.mol_coulomb_energy
					self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
			self.aux_func = self.lagrange_mult1 * (tf.reduce_sum(self.replace_scalars) - 1.0)
			self.zero_bound_func = self.lagrange_mult2 * (self.replace_scalars - tf.square(self.zero_bound))
			self.one_bound_func = self.lagrange_mult3 * (1.0 - self.replace_scalars - tf.square(self.one_bound))
			self.lagrangian = self.total_energy + self.aux_func + self.zero_bound_func + self.one_bound_func
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
				# self.scalars_energy_grad = tf.gradients(self.total_energy, self.replace_scalars)[0]
				# self.scalars_aux_grad, self.lagrange_grad = tf.gradients(self.aux_func,
				# 	[self.replace_scalars, self.lagrange_mult])
				# self.scalars_energy_loss = self.loss_op(self.scalars_energy_grad)
				# self.scalars_aux_loss = self.loss_op(self.scalars_aux_grad)
				# self.lagrange_loss = self.loss_op(self.lagrange_grad)
				(self.scalars_grad, self.lagrange_grad1, self.lagrange_grad2, self.lagrange_grad3,
					self.zero_grad, self.one_grad) = tf.gradients(self.lagrangian,
					[self.replace_scalars, self.lagrange_mult1, self.lagrange_mult2, self.lagrange_mult3,
					self.zero_bound, self.one_bound])
				self.scalars_loss = self.loss_op(self.scalars_grad)
				self.lagrange1_loss = self.loss_op(self.lagrange_grad1)
				self.lagrange2_loss = self.loss_op(self.lagrange_grad2)
				self.lagrange3_loss = self.loss_op(self.lagrange_grad3)
				self.zero_loss = self.loss_op(self.zero_grad)
				self.one_loss = self.loss_op(self.one_grad)
				self.total_loss = (self.scalars_loss + (self.lagrange1_loss
					+ self.lagrange2_loss + self.lagrange3_loss + self.zero_loss + self.one_loss))

			self.minimize_op = self.optimizer(self.total_loss, 10.0 * self.learning_rate,
				self.momentum, [self.replace_scalars, self.lagrange_mult1, self.lagrange_mult2,
				self.lagrange_mult3, self.zero_bound, self.one_bound])

			vars_in_checkpoint = tf.train.list_variables(self.network_directory)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
			self.sess.run(tf.variables_initializer(all_variables))
			restore_vars = [v for v in all_variables if ("lagrange_mult" not in v.name) and ("replace_scalars" not in v.name) and ("bound" not in v.name)]
			self.saver = tf.train.Saver(var_list=restore_vars)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def compute_normalization(self):
		elements = self.mol_set.AtomTypes().tolist()
		avg_energy = {element:0. for element in elements}
		num_elements = len(avg_energy)
		num_elements_dense = max(elements)+1
		regres_energies = np.zeros(self.num_molecules)
		# Use dense zero-padded arrays to avoid index logic.
		stoichs = np.zeros((self.num_molecules, num_elements_dense))

		srange_inner = 4.0*1.889725989
		srange_outer = 6.5*1.889725989
		lrange_inner = 13.0*1.889725989
		lrange_outer = 15.0*1.889725989
		a, b, c, d, e, f, g, h = -9.83315, 4.49307, -0.784438, 0.0747019, -0.00419095, 0.000138593, -2.50374e-6, 1.90818e-8

		for i, mol in enumerate(self.mol_set.mols):
			unique, counts = np.unique(mol.atoms, return_counts=True)
			stoich = np.zeros(num_elements_dense)
			for j in range(len(unique)):
				stoich[unique[j]] = counts[j]
			stoichs[i] += stoich
			if self.train_charges:
				mol.BuildDistanceMatrix()
				mol.coul_matrix = mol.properties["charges"] * np.expand_dims(mol.properties["charges"], axis=-1)
				for j in range(mol.NAtoms()):
					mol.DistMatrix[j,j] = 1.0
					mol.coul_matrix[j,j] = 0.0
				dist = mol.DistMatrix * 1.889725989
				dist = np.where(np.less(dist, srange_inner), np.ones_like(dist) * srange_inner, dist)
				dist = np.where(np.greater(dist, lrange_outer), np.ones_like(dist) * lrange_outer, dist)
				dist2 = dist * dist
				dist3 = dist2 * dist
				dist4 = dist3 * dist
				dist5 = dist4 * dist
				dist6 = dist5 * dist
				dist7 = dist6 * dist
				kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
				mrange_energy = np.sum(kern * mol.coul_matrix, axis=1)
				lrange_energy = np.sum(mol.coul_matrix, axis=1) / lrange_outer
				coulomb_energy = np.sum((mrange_energy - lrange_energy) / 2.0)
				regres_energies[i] = mol.properties["energy"] - coulomb_energy
			else:
				regres_energies[i] = mol.properties["energy"]
		noa = np.sum(stoichs, axis=0)
		x,r = np.linalg.lstsq(stoichs,regres_energies)[:2]
		for element in elements:
			avg_energy[element] = x[element]

		self.energy_fit = np.zeros((self.mol_set.max_atomic_num()+1))
		for element in avg_energy.keys():
			self.energy_fit[element] = avg_energy[element]
		energies = self.energy_data - np.sum(self.energy_fit[self.Z_data], axis=1)
		self.energy_mean = np.mean(energies)
		self.energy_std = np.std(energies)
		print("---- Results of Stoichiometric Model ----")
		print("MeanE  Energy: ", np.mean(energies))
		print("StdE  Energy: ", np.std(energies))
		print("MXE  Energy: ", np.max(energies))
		print("MNE  Energy: ", np.min(energies))
		print("RMSE Energy: ", np.sqrt(np.average(np.square(energies))))
		print("AvE: ", avg_energy)

		if self.train_charges:
			self.charge_mean = np.zeros((self.mol_set.max_atomic_num()+1))
			self.charge_std = np.zeros((self.mol_set.max_atomic_num()+1))
			for element in self.elements:
				element_idxs = np.where(np.equal(self.Z_data, element))
				element_charges = self.charges_data[element_idxs]
				self.charge_mean[element] = np.mean(element_charges)
				self.charge_std[element] = np.std(element_charges)
		self.embed_shape = self.codes_shape * (self.radial_rs.shape[0] + self.angular_rs.shape[0] * self.theta_s.shape[0])
		self.label_shape = 1
		return

	def calculate_coulomb_energy(self, dxyzs, q1q2, scatter_idx):
		"""
		Polynomial cutoff 1/r (in BOHR) obeying:
		kern = 1/r at SROuter and LRInner
		d(kern) = d(1/r) (true force) at SROuter,LRInner
		d**2(kern) = d**2(1/r) at SROuter and LRInner.
		d(kern) = 0 (no force) at/beyond SRInner and LROuter

		The hard cutoff is LROuter
		"""
		srange_inner = tf.constant(4.0*1.889725989, dtype=self.tf_precision)
		srange_outer = tf.constant(6.5*1.889725989, dtype=self.tf_precision)
		lrange_inner = tf.constant(13.0*1.889725989, dtype=self.tf_precision)
		lrange_outer = tf.constant(15.0*1.889725989, dtype=self.tf_precision)
		a, b, c, d, e, f, g, h = -9.83315, 4.49307, -0.784438, 0.0747019, -0.00419095, 0.000138593, -2.50374e-6, 1.90818e-8
		dist = tf.norm(dxyzs+1.e-16, axis=-1)
		dist *= 1.889725989
		dist = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) * srange_inner, dist)
		dist = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) * lrange_outer, dist)
		dist2 = dist * dist
		dist3 = dist2 * dist
		dist4 = dist3 * dist
		dist5 = dist4 * dist
		dist6 = dist5 * dist
		dist7 = dist6 * dist
		kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
		mrange_energy = tf.reduce_sum(kern * q1q2, axis=1)
		lrange_energy = tf.reduce_sum(q1q2, axis=1) / lrange_outer
		coulomb_energy = (mrange_energy - lrange_energy) / 2.0
		return tf.reduce_sum(tf.scatter_nd(scatter_idx, coulomb_energy, [self.batch_size, self.max_num_atoms]), axis=-1)

	def evaluate_set(self, mset):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mset.MaxNAtom()
			self.batch_size = 100
			self.eval_prepare()
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			energy_data[i] = mol.properties["energy"]
			gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0
		energy_true, energy_pred = [], []
		gradients_true, gradient_preds = [], []
		charges_true, charge_preds = [], []
		for ministep in range(int(num_mols / self.batch_size)):
			eval_pointer += self.batch_size
			batch_xyzs = xyz_data[eval_pointer - self.batch_size:eval_pointer]
			batch_Zs = Z_data[eval_pointer - self.batch_size:eval_pointer]
			nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
			batch_data = []
			batch_data.append(batch_xyzs)
			batch_data.append(batch_Zs)
			batch_data.append(nn_pairs)
			batch_data.append(nn_triples)
			batch_data.append(coulomb_pairs)
			batch_data.append(num_atoms_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(energy_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(gradient_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(charges_data[eval_pointer - self.batch_size:eval_pointer])
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_charges:
				total_energy, energy_label, gradients, gradient_labels, charges, charge_labels = self.sess.run([self.total_energy,
					self.energy_pl, self.gradients, self.gradient_labels, self.charges, self.charge_labels], feed_dict=feed_dict)
				charges_true.append(charge_labels)
				charge_preds.append(charges)
			else:
				total_energy, energy_label, gradients, gradient_labels = self.sess.run([self.total_energy,
					self.energy_pl, self.gradients, self.gradient_labels], feed_dict=feed_dict)
			energy_true.append(energy_label)
			energy_pred.append(total_energy)
			gradients_true.append(gradient_labels)
			gradient_preds.append(gradients)
		energy_true = np.concatenate(energy_true)
		energy_pred = np.concatenate(energy_pred)
		gradients_true = np.concatenate(gradients_true)
		gradient_preds = np.concatenate(gradient_preds)
		energy_errors = energy_true - energy_pred
		gradient_errors = gradients_true - gradient_preds
		if self.train_charges:
			charges_true = np.concatenate(charges_true)
			charge_preds = np.concatenate(charge_preds)
			charge_errors = charges_true - charge_preds
			return energy_errors, gradient_errors, charge_errors
		else:
			return energy_errors, gradient_errors

	def get_atom_energies(self, mset):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mset.MaxNAtom()
			self.batch_size = 100
			self.eval_prepare()
		atom_energy_data = []
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			energy_data[i] = mol.properties["energy"]
			gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0
		# energy_true, energy_pred = [], []
		# gradients_true, gradient_preds = [], []
		# charges_true, charge_preds = [], []
		for ministep in range(int(num_mols / self.batch_size)):
			eval_pointer += self.batch_size
			batch_xyzs = xyz_data[eval_pointer - self.batch_size:eval_pointer]
			batch_Zs = Z_data[eval_pointer - self.batch_size:eval_pointer]
			nn_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
			batch_data = []
			batch_data.append(batch_xyzs)
			batch_data.append(batch_Zs)
			batch_data.append(nn_pairs)
			batch_data.append(nn_triples)
			batch_data.append(coulomb_pairs)
			batch_data.append(num_atoms_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(energy_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(gradient_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(charges_data[eval_pointer - self.batch_size:eval_pointer])
			feed_dict = self.fill_feed_dict(batch_data)
			batch_atom_energies = self.sess.run(self.atom_nn_energy_tmp,
				feed_dict=feed_dict)
			atom_energy_data.append(batch_atom_energies)
		atom_energy_data = np.concatenate(atom_energy_data)
		Z_data_tmp = Z_data[:np.shape(atom_energy_data)[0]]
		elements = mset.AtomTypes().tolist()
		element_energy_data = []
		for element in elements:
			element_idxs = np.where(np.equal(Z_data_tmp, element))
			atom_energies = atom_energy_data[element_idxs]
			element_energy_data.append(atom_energies)
		return element_energy_data

	def get_energy_force_function(self,mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.eval_prepare()
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		num_atoms_data[0] = mol.NAtoms()
		Z_data[0][:mol.NAtoms()] = mol.atoms
		def EF(xyz_, DoForce=True):
			xyz_data[0][:mol.NAtoms()] = xyz_
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 15.0, self.max_num_atoms, True, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs, self.num_atoms_pl:num_atoms_data}
			if (DoForce):
				energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
				return energy[0], -JOULEPERHARTREE*gradients
			else:
				energy = self.sess.run(self.total_energy, feed_dict=feed_dict)
				return energy[0]
		return EF

	def evaluate_alchem_mol(self, mols):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = max([mol.NAtoms() for mol in mols])
			self.batch_size = 1
			self.alchem_prepare()
		xyz_data = np.zeros((len(mols), self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((len(mols), self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((len(mols)), dtype = np.int32)
		max_alchem_atoms = np.argmax(num_atoms_data)
		def alchem_energy_force(mols, delta, return_forces=True):
			for i, mol in enumerate(mols):
				xyz_data[i][:mol.NAtoms()] = mols[i].coords
				Z_data[i][:mol.NAtoms()] = mols[i].atoms
				num_atoms_data[i] = mol.NAtoms()
			nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = MolEmb.Make_NLTensor(xyz_data[max_alchem_atoms:max_alchem_atoms+1],
							Z_data[max_alchem_atoms:max_alchem_atoms+1], 15.0, self.max_num_atoms, True, False)
			delta = np.array(delta).reshape(1)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs,
						self.num_atoms_pl:num_atoms_data, self.delta_pl:delta}
			energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
			return energy[0], -gradients
		return alchem_energy_force

	def element_opt(self, mol, replace_idx):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.element_opt_prepare()
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		Z_data[0][:mol.NAtoms()] = mol.atoms
		num_atoms_data[0] = mol.NAtoms()
		replace_idx_data = np.array(replace_idx, dtype=np.int32)
		xyz_data[0][:mol.NAtoms()] = mol.coords
		nn_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = MolEmb.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = MolEmb.Make_NLTensor(xyz_data, Z_data, 15.0, self.max_num_atoms, True, False)
		total_loss = 10.0
		feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
					self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs,
					self.num_atoms_pl:num_atoms_data, self.replace_idx_pl:replace_idx_data}
		while total_loss > 0.0005:
			(_, total_loss, scalars_loss, lagrange1_loss, lagrange2_loss, zero_loss, energy,
				replace_scalars, lagrange_mult1, lagrange_mult2, zero_bound) = self.sess.run([self.minimize_op, self.total_loss,
				self.scalars_loss, self.lagrange1_loss, self.lagrange2_loss, self.zero_loss,
				self.total_energy, self.replace_scalars, self.lagrange_mult1, self.lagrange_mult2,
				self.zero_bound], feed_dict=feed_dict)
			LOGGER.info("Total loss: %11.8f  Scalars loss: %11.8f  Lagrange1 loss: %11.8f  Lagrange2 loss: %11.8f  Zero loss: %11.8f",
				total_loss, scalars_loss, lagrange1_loss, lagrange2_loss, zero_loss)
			LOGGER.info("Atomization Energy: %11.8f  Lagrange Multiplier: %11.8f  Sum of Scalars: %11.8f",
				energy, lagrange_mult1, np.sum(replace_scalars))
		print(replace_scalars)
		return

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
			code_kernel1 = tf.get_variable(name="CodeKernel1", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
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
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 2],
					stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([2], dtype=self.tf_precision), name='biases')
				outputs = tf.matmul(activations, weights) + biases
				variables.append(weights)
				variables.append(biases)
				atom_nn_energy = tf.scatter_nd(indices, outputs[...,0], [self.batch_size, self.max_num_atoms])
				atom_nn_charge = tf.scatter_nd(indices, outputs[...,1], [self.batch_size, self.max_num_atoms])
			return atom_nn_energy, atom_nn_charge, variables
