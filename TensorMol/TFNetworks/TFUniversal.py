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
		self.train_dipole = PARAMS["train_dipole"]
		self.train_quadrupole = PARAMS["train_quadrupole"]
		self.train_sparse = PARAMS["train_sparse"]
		self.sparse_cutoff = PARAMS["sparse_cutoff"]
		self.profiling = PARAMS["Profiling"]
		self.activation_function_type = PARAMS["NeuronType"]
		self.test_ratio = PARAMS["TestRatio"]
		self.element_codes = ELEMENTCODES
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
		if self.train_sparse:
			self.max_num_pairs = self.mol_set.max_neighbors()
		self.step = 0
		self.test_freq = PARAMS["test_freq"]
		self.network_type = "GauSH_Univ"
		self.name = self.network_type+"_"+self.mol_set_name+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.network_directory = PARAMS["networks_directory"]+self.name
		self.l_max = PARAMS["SH_LMAX"]
		self.gaussian_params = PARAMS["RBFS"]

		LOGGER.info("learning rate: %f", self.learning_rate)
		LOGGER.info("batch size:    %d", self.batch_size)
		LOGGER.info("max steps:     %d", self.max_steps)
		return

	def __getstate__(self):
		state = self.__dict__.copy()
		remove_vars = ["mol_set", "activation_function", "xyz_data", "Z_data", "energy_data", "dipole_data",
						"num_atoms_data", "gradient_data", "pairs_data"]
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
		self.theta_s = 2.0 * np.pi * np.linspace(0, (num_angular_theta_s - 1.0) / num_angular_theta_s, num_angular_theta_s)
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
			self.energy_train_step(self.step)
			if self.step%self.test_freq==0:
				test_loss = self.energy_test_step(self.step)
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
		pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
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
		# self.mulliken_charges_data = np.zeros((self.num_molecules, self.max_num_atoms), dtype = np.float64)
		self.num_atoms_data = np.zeros((self.num_molecules), dtype = np.int32)
		self.energy_data = np.zeros((self.num_molecules), dtype = np.float64)
		self.gradient_data = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype=np.float64)
		self.nearest_neighbors_data = np.zeros((self.num_molecules, self.max_num_atoms, 2), dtype=np.int32)
		if self.train_dipole:
			self.dipole_data = np.zeros((self.num_molecules, 3), dtype = np.float64)
		if self.train_sparse:
			self.pairs_data = np.zeros((self.num_molecules, self.max_num_atoms, self.max_num_pairs, 2), dtype=np.uint16)
		for i, mol in enumerate(self.mol_set.mols):
			self.xyz_data[i][:mol.NAtoms()] = mol.coords
			self.Z_data[i][:mol.NAtoms()] = mol.atoms
			# self.mulliken_charges_data[i][:mol.NAtoms()] = mol.properties["mulliken_charges"]
			self.energy_data[i] = mol.properties["atomization"]
			self.gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			self.nearest_neighbors_data[i][:mol.NAtoms()] = mol.nearest_ns
			if self.train_dipole:
				self.dipole_data[i] = mol.properties["dipole"]
			if self.train_quadrupole:
				self.quadrupole_data[i] = mol.properties["quadrupole"]
			if self.train_sparse:
				for j, atom_pairs in enumerate(mol.neighbor_list):
					self.pairs_data[i,j,:len(atom_pairs)] = np.stack([np.array(mol.neighbor_list[j]), mol.atoms[atom_pairs]], axis=-1)
				self.batch_mol_idxs = np.tile(np.arange(self.batch_size), (1, self.max_num_pairs, self.max_num_atoms, 1)).T
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

	def get_energy_train_batch(self, batch_size):
		if self.train_pointer + batch_size >= self.num_train_cases:
			np.random.shuffle(self.train_idxs)
			self.train_pointer = 0
		self.train_pointer += batch_size
		batch_data = []
		batch_data.append(self.xyz_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.Z_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.num_atoms_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.energy_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.gradient_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		# batch_data.append(self.nearest_neighbors_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		NL = NeighborListSet(self.xyz_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]], self.num_atoms_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]], True, True, self.Z_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]], sort_=True)
		rad_p_ele, ang_t_elep, mil_j, mil_jk = NL.buildPairsAndTriplesWithEleIndexLinear(self.radial_cutoff, self.angular_cutoff, self.elements, self.element_pairs)
		batch_data.append(rad_p_ele)
		batch_data.append(ang_t_elep)
		batch_data.append(mil_j)
		batch_data.append(mil_jk)
		if self.train_sparse:
			pair_batch_data = np.concatenate((self.batch_mol_idxs, self.pairs_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]]), axis=-1)
			batch_data.append(pair_batch_data)
		return batch_data

	def get_energy_test_batch(self, batch_size):
		if self.test_pointer + batch_size >= self.num_test_cases:
			self.test_pointer = 0
		self.test_pointer += batch_size
		batch_data = []
		batch_data.append(self.xyz_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.Z_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.num_atoms_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.energy_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.gradient_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		# batch_data.append(self.nearest_neighbors_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		NL = NeighborListSet(self.xyz_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]], self.num_atoms_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]], True, True, self.Z_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]], sort_=True)
		rad_p_ele, ang_t_elep, mil_j, mil_jk = NL.buildPairsAndTriplesWithEleIndexLinear(self.radial_cutoff, self.angular_cutoff, self.elements, self.element_pairs)
		batch_data.append(rad_p_ele)
		batch_data.append(ang_t_elep)
		batch_data.append(mil_j)
		batch_data.append(mil_jk)
		if self.train_sparse:
			pair_batch_data = np.concatenate((self.batch_mol_idxs, self.pairs_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]]), axis=-1)
			batch_data.append(pair_batch_data)
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

	def fill_energy_feed_dict(self, batch_data):
		"""
		Fill the tensorflow feed dictionary.

		Args:
			batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
			and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

		Returns:
			Filled feed dictionary.
		"""
		pl_list = [self.xyzs_pl, self.Zs_pl, self.num_atoms_pl, self.energy_pl, self.gradients_pl]#, self.nearest_neighbors_pl]
		pl_list.append(self.Radp_Ele_pl)
		pl_list.append(self.Angt_Elep_pl)
		pl_list.append(self.mil_j_pl)
		pl_list.append(self.mil_jk_pl)
		if self.train_sparse:
			pl_list.append(self.pairs_pl)
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
		output = tf.zeros([self.batch_size, self.max_num_atoms], dtype=self.tf_precision)
		with tf.variable_scope("energy_network", reuse=tf.AUTO_REUSE):
			code_kernel1 = tf.get_variable(name="CodeKernel", shape=(4, 4),dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(4, 4),dtype=self.tf_precision)
			coded_weights = tf.matmul(atom_codes, code_kernel1)
			coded_embed = tf.einsum('ikj,ij->ikj', embed, coded_weights)
			coded_embed = tf.reshape(tf.einsum('ikj,jl->ikl', coded_embed, code_kernel2), [tf.shape(embed)[0], -1])
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
				output += tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
				tf.verify_tensor_all_finite(output,"Nan in output!!!")
		return output, variables

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

	def energy_train_step(self, step):
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
		num_mols = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data = self.get_energy_train_batch(self.batch_size)
			feed_dict = self.fill_energy_feed_dict(batch_data)
			if self.train_gradients:
				_, summaries, total_loss, energy_loss, gradient_loss = self.sess.run([self.energy_train_op,
				self.summary_op, self.energy_losses, self.energy_loss, self.gradient_loss], feed_dict=feed_dict)
				train_gradient_loss += gradient_loss
			else:
				if self.profiling:
					_, summaries, total_loss, energy_loss = self.sess.run([self.energy_train_op,
					self.summary_op, self.energy_losses, self.energy_loss], feed_dict=feed_dict,
					options=self.options, run_metadata=self.run_metadata)
					fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
					chrome_trace = fetched_timeline.generate_chrome_trace_format()
					with open('timeline_step_%d.json' % ministep, 'w') as f:
						f.write(chrome_trace)
				else:
					_, summaries, total_loss, energy_loss = self.sess.run([self.energy_train_op,
					self.summary_op, self.energy_losses, self.energy_loss], feed_dict=feed_dict)
			train_loss += total_loss
			train_energy_loss += energy_loss
			num_mols += self.batch_size
			self.summary_writer.add_summary(summaries, step * int(Ncase_train/self.batch_size) + ministep)
		duration = time.time() - start_time
		self.print_epoch(step, duration, train_loss, train_energy_loss, train_gradient_loss)
		return

	def energy_test_step(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		test_loss =  0.0
		start_time = time.time()
		Ncase_test = self.num_test_cases
		num_mols = 0
		test_energy_loss = 0.0
		test_gradient_loss = 0.0
		test_charge_loss = 0.0
		test_epoch_energy_labels, test_epoch_energy_outputs = [], []
		test_epoch_force_labels, test_epoch_force_outputs = [], []
		num_atoms_epoch = []
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.get_energy_test_batch(self.batch_size)
			feed_dict = self.fill_energy_feed_dict(batch_data)
			total_energies, energy_labels, gradients, gradient_labels, total_loss, energy_loss, gradient_loss, num_atoms, gaussian_params = self.sess.run([self.total_energy,
			self.energy_pl, self.gradients, self.gradient_labels, self.energy_losses, self.energy_loss,
			self.gradient_loss, self.num_atoms_pl, self.gaussian_params],  feed_dict=feed_dict)
			test_loss += total_loss
			num_mols += self.batch_size
			test_energy_loss += energy_loss
			test_gradient_loss += gradient_loss
			test_epoch_energy_labels.append(energy_labels)
			test_epoch_energy_outputs.append(total_energies)
			test_epoch_force_labels.append(-1.0 * gradient_labels)
			test_epoch_force_outputs.append(-1.0 * gradients)
			num_atoms_epoch.append(num_atoms)
		test_epoch_energy_labels = np.concatenate(test_epoch_energy_labels)
		test_epoch_energy_outputs = np.concatenate(test_epoch_energy_outputs)
		test_epoch_energy_errors = test_epoch_energy_labels - test_epoch_energy_outputs
		test_epoch_force_labels = np.concatenate(test_epoch_force_labels)
		test_epoch_force_outputs = np.concatenate(test_epoch_force_outputs)
		test_epoch_force_errors = test_epoch_force_labels - test_epoch_force_outputs
		num_atoms_epoch = np.sum(np.concatenate(num_atoms_epoch))
		duration = time.time() - start_time
		for i in [random.randint(0, self.batch_size - 1) for _ in xrange(20)]:
			LOGGER.info("Energy label: %11.8f  Energy output: %11.8f", test_epoch_energy_labels[i], test_epoch_energy_outputs[i])
		for i in [random.randint(0, num_atoms_epoch - 1) for _ in xrange(20)]:
			LOGGER.info("Forces label: %s  Forces output: %s", test_epoch_force_labels[i], test_epoch_force_outputs[i])
		LOGGER.info("MAE  Energy: %11.8f  Forces: %11.8f", np.mean(np.abs(test_epoch_energy_errors)),
		np.mean(np.abs(test_epoch_force_errors)))
		LOGGER.info("MSE  Energy: %11.8f  Forces: %11.8f", np.mean(test_epoch_energy_errors),
		np.mean(test_epoch_force_errors))
		LOGGER.info("RMSE Energy: %11.8f  Forces: %11.8f", np.sqrt(np.mean(np.square(test_epoch_energy_errors))),
		np.sqrt(np.mean(np.square(test_epoch_force_errors))))
		LOGGER.info("Gaussian paramaters: %s", gaussian_params)
		self.print_epoch(step, duration, test_loss, test_energy_loss, test_gradient_loss, testing=True)
		return test_loss

	def compute_normalization(self):
		self.energy_mean = np.mean(self.energy_data)
		self.energy_stddev = np.std(self.energy_data)
		self.embed_shape = 4 * self.gaussian_params.shape[0] * (self.l_max + 1) ** 2
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
			self.energy_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size])
			self.gradients_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
			self.dipole_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, 3])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
			# self.nearest_neighbors_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, 2])
			self.Radp_Ele_pl=tf.placeholder(tf.int32, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int32, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int32, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int32, shape=tuple([None,4]))
			if self.train_sparse:
				self.pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, self.max_num_pairs, 3])

			elements = tf.constant(self.elements, dtype = tf.int32)
			element_pairs = tf.constant(self.element_pairs, dtype = tf.int32)
			radial_rs = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_rs = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			theta_s = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=False, dtype=self.tf_precision)
			energy_mean = tf.Variable(self.energy_mean, trainable=False, dtype = self.tf_precision)
			energy_stddev = tf.Variable(self.energy_stddev, trainable=False, dtype = self.tf_precision)
			with tf.name_scope('embedding'):
				self.Scatter_Sym = TFSymSet_Scattered_Linear_WithEle_Channel_Multi(self.xyzs_pl, self.Zs_pl, elements, radial_rs, radial_cutoff, element_pairs, angular_rs, zeta, eta, angular_cutoff, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl, self.element_codes)
				# if self.train_sparse:
				# 	dxyzs, pair_Zs = sparsify_coords(self.xyzs_pl, self.Zs_pl, self.pairs_pl)
				# 	canon_xyzs = gs_canonicalizev2(dxyzs, self.Zs_pl)
				# 	embed, mol_idx = tf_sparse_gaush_element_channel(canon_xyzs, self.Zs_pl, pair_Zs,
				# 					elements, self.gaussian_params, self.l_max)
				# else:
				# 	dxyzs, padding_mask = center_dxyzs(self.xyzs_pl, self.Zs_pl)
				# 	nearest_neighbors = tf.gather_nd(self.nearest_neighbors_pl, padding_mask)
				# 	canon_xyzs, perm_canon_xyzs = gs_canonicalize(dxyzs, nearest_neighbors)
				# 	embed = tf_gaush_embed_channel(canon_xyzs, self.Zs_pl,
				# 					elements, self.gaussian_params, self.l_max, self.element_codes)
				# 	perm_embed = tf_gaush_embed_channel(perm_canon_xyzs, self.Zs_pl,
				# 							elements, self.gaussian_params, self.l_max, self.element_codes)
				# 	atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				atom_energies, energy_variables = self.energy_inference(embed, atom_codes, padding_mask)
				perm_atom_energies, _ = self.energy_inference(perm_embed, atom_codes, padding_mask)
				norm_bp_energy = ((tf.reshape(tf.reduce_sum(atom_energies, axis=1), [self.batch_size])
								+ tf.reshape(tf.reduce_sum(perm_atom_energies, axis=1), [self.batch_size])) / 2.0)
				self.bp_energy = (norm_bp_energy * energy_stddev) + energy_mean
				self.total_energy = self.bp_energy
				self.energy_loss = self.loss_op(self.total_energy - self.energy_pl) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
				tf.summary.scalar("energy loss", self.energy_loss)
				tf.add_to_collection('energy_losses', self.energy_loss)
			with tf.name_scope('gradients'):
				xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(xyz_grad, padding_mask)
				self.gradient_labels = tf.gather_nd(self.gradients_pl, padding_mask)
				self.gradient_loss = 0.001 * self.loss_op(self.gradients - self.gradient_labels) / (3 * tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision))
				if self.train_gradients:
					tf.add_to_collection('energy_losses', self.gradient_loss)
					tf.summary.scalar("gradient loss", self.gradient_loss)
			self.energy_losses = tf.add_n(tf.get_collection('energy_losses'))
			tf.summary.scalar("energy losses", self.energy_losses)

			self.energy_train_op = self.optimizer(self.energy_losses, self.learning_rate, self.momentum)
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

	def print_epoch(self, step, duration, loss, energy_loss, gradient_loss, testing=False):
		if testing:
			LOGGER.info("step: %5d  duration: %.3f  test loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
			step, duration, loss, energy_loss, gradient_loss)
		else:
			LOGGER.info("step: %5d  duration: %.3f  train loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
			step, duration, loss, energy_loss, gradient_loss)
		return
