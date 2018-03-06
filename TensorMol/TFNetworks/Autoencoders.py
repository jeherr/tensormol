"""
Autoencoder for rotational invariance of GauSH
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import random
import sys
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle

from ..TFDescriptors.RawSH import *

class GauSHEncoder(object):
	"""
	Class for training rotational invariance of the GauSH
	descriptor.
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
		self.train_rotation = PARAMS["train_rotation"]
		self.train_sparse = PARAMS["train_sparse"]
		self.sparse_cutoff = PARAMS["sparse_cutoff"]
		self.profiling = PARAMS["Profiling"]
		self.activation_function_type = PARAMS["NeuronType"]
		self.randomize_data = PARAMS["RandomizeData"]
		self.test_ratio = PARAMS["TestRatio"]
		self.assign_activation()

		self.coulomb_cutoff = PARAMS["EECutoffOff"]
		self.dsf_alpha = PARAMS["DSFAlpha"]
		self.elu_width = PARAMS["Elu_Width"]

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
		self.max_num_atoms = self.mol_set.MaxNAtoms()
		self.num_molecules = len(self.mol_set.mols)
		if self.train_sparse:
			for mol in self.mol_set.mols:
				mol.make_neighbors(self.sparse_cutoff)
			self.max_num_pairs = self.mol_set.max_neighbors()

		LOGGER.info("learning rate: %f", self.learning_rate)
		LOGGER.info("batch size:    %d", self.batch_size)
		LOGGER.info("max steps:     %d", self.max_steps)

		self.network_type = "GauSHAE"
		self.name = self.network_type+"_"+self.mol_set_name+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.network_directory = PARAMS["networks_directory"]+self.name
		self.l_max = PARAMS["SH_LMAX"]
		self.gaussian_params = PARAMS["RBFS"]
		return

	def __getstate__(self):
		state = self.__dict__.copy()
		remove_vars = ["mol_set", "activation_function", "xyz_data", "Z_data"]
		for var in remove_vars:
			try:
				del state[var]
			except:
				pass
		return state

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

	def sigmoid_with_param(self, x):
		return tf.log(1.0+tf.exp(tf.multiply(tf.cast(PARAMS["sigmoid_alpha"], dtype=self.tf_precision), x)))/tf.cast(PARAMS["sigmoid_alpha"], dtype=self.tf_precision)

	def start_training(self):
		self.load_data_to_scratch()
		self.compute_normalization()
		self.save_network()
		self.train_prepare()
		self.train()

	def restart_training(self):
		self.reload_set()
		self.load_data_to_scratch()
		self.train_prepare(restart=True)
		self.train()

	def train(self):
		test_freq = PARAMS["test_freq"]
		mini_test_loss = 1e10
		for step in range(1, self.max_steps+1):
			self.train_step(step)
			if step%test_freq==0:
				test_loss = self.test_step(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_checkpoint(step)
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
		# import TensorMol.PickleTM
		# network_member_variables = TensorMol.PickleTM.UnPickleTM(f)
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
		self.num_atoms_data = np.zeros((self.num_molecules), dtype = np.int32)
		if self.train_sparse:
			self.pairs_data = np.zeros((self.num_molecules, self.max_num_atoms, self.max_num_pairs, 4), dtype=np.uint16)
		for i, mol in enumerate(self.mol_set.mols):
			self.xyz_data[i][:mol.NAtoms()] = mol.coords
			self.Z_data[i][:mol.NAtoms()] = mol.atoms
			if self.train_sparse:
				for j, atom_pairs in enumerate(mol.neighbor_list):
					self.pairs_data[i,j,:len(atom_pairs)] = np.stack([np.array([i for _ in range(len(atom_pairs))]),
						np.array([j for _ in range(len(atom_pairs))]), np.array(atom_pairs), mol.atoms[atom_pairs]]).T
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
		batch_data = []
		batch_data.append(self.xyz_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.Z_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		batch_data.append(self.num_atoms_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		if self.train_sparse:
			batch_data.append(self.pairs_data[self.train_idxs[self.train_pointer - batch_size:self.train_pointer]])
		return batch_data

	def get_test_batch(self, batch_size):
		if self.test_pointer + batch_size >= self.num_test_cases:
			self.test_pointer = 0
		self.test_pointer += batch_size
		batch_data = []
		batch_data.append(self.xyz_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.Z_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		batch_data.append(self.num_atoms_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
		if self.train_sparse:
			batch_data.append(self.pairs_data[self.test_idxs[self.test_pointer - batch_size:self.test_pointer]])
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
			tf.add_to_collection('losses', weightdecay)
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
		pl_list = [self.xyzs_pl, self.Zs_pl, self.num_atoms_pl]
		if self.train_sparse:
			pl_list.append(self.pairs_pl)
		feed_dict={i: d for i, d in zip(pl_list, batch_data)}
		return feed_dict

	def print_epoch(self, step, duration, loss, embed_loss, rotation_loss, num_atoms, testing=False):
		if testing:
			LOGGER.info("step: %5d  duration: %.3f  test loss: %.10f  embed loss: %.10f  rotation loss: %.10f",
			step, duration, loss / num_atoms, embed_loss / num_atoms, rotation_loss / num_atoms)
		else:
			LOGGER.info("step: %5d  duration: %.3f  train loss: %.10f  embed loss: %.10f  rotation loss: %.10f",
			step, duration, loss / num_atoms, embed_loss / num_atoms, rotation_loss / num_atoms)
		return

	def encoder(self, inputs):
		"""
		Builds an encoder for the GauSH descriptor

		Args:
			inputs (tf.float): NCase x Embed Shape of the embeddings to encode
		Returns:
			The latent space vectors
		"""
		with tf.name_scope("encoder"):
			if len(self.hidden_layers) == 0:
				with tf.name_scope('encoder_hidden'):
					weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.latent_shape],
							stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([self.latent_shape], dtype=self.tf_precision), name='biases')
					latent_vectors = self.activation_function(tf.matmul(inputs, weights) + biases)
			else:
				for i in range(len(self.hidden_layers)):
					if i == 0:
						with tf.name_scope('encoder_hidden1'):
							weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[i]],
									stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
							hidden_output = self.activation_function(tf.matmul(inputs, weights) + biases)
					else:
						with tf.name_scope('encoder_hidden'+str(i+1)):
							weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
									stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
							hidden_output = self.activation_function(tf.matmul(hidden_output, weights) + biases)
				with tf.name_scope('latent_space'):
					weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], self.latent_shape],
							stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([self.latent_shape], dtype=self.tf_precision), name='biases')
					latent_vectors = tf.matmul(hidden_output, weights) + biases
		return latent_vectors

	def decoder(self, latent_vector):
		"""
		Builds a decoder for the GauSH descriptor

		Args:
			inputs (tf.float): NCase x latent vector shape tensor of the embedding cases
		Returns:
			The decoded GauSH descriptor
		"""
		with tf.name_scope("decoder"):
			if len(self.hidden_layers) == 0:
				with tf.name_scope('encoder_hidden'):
					weights = self.variable_with_weight_decay(shape=[self.latent_shape, self.embed_shape],
							stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([self.embed_shape], dtype=self.tf_precision), name='biases')
					outputs = self.activation_function(tf.matmul(latent_vector, weights) + biases)
			else:
				for i in range(len(self.hidden_layers)):
					if i == 0:
						with tf.name_scope('decoder_hidden1'):
							weights = self.variable_with_weight_decay(shape=[self.latent_shape, self.hidden_layers[-1]],
									stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[-1]], dtype=self.tf_precision), name='biases')
							hidden_output = self.activation_function(tf.matmul(latent_vector, weights) + biases)
					else:
						with tf.name_scope('decoder_hidden'+str(i+1)):
							weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-i], self.hidden_layers[-1-i]],
									stddev=math.sqrt(2.0 / float(self.hidden_layers[-i])), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[-1-i]], dtype=self.tf_precision), name='biases')
							hidden_output = self.activation_function(tf.matmul(hidden_output, weights) + biases)
				with tf.name_scope('decoder_output'):
					weights = self.variable_with_weight_decay(shape=[self.hidden_layers[0], self.embed_shape],
							stddev=math.sqrt(2.0 / float(self.hidden_layers[0])), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([self.embed_shape], dtype=self.tf_precision), name='biases')
					outputs = tf.matmul(hidden_output, weights) + biases
		return outputs

	def optimizer(self, loss, learning_rate, momentum):
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
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def loss_op(self, error):
		loss = tf.nn.l2_loss(error)
		return loss

	def compute_normalization(self):
		# xyzs_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
		# Zs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms])
		# if self.train_sparse:
		# 	pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, self.max_num_pairs, 4])
		# gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_precision)
		# elements = tf.constant(self.elements, dtype = tf.int32)
		#
		# rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size, self.max_num_atoms], maxval=2.0, dtype=self.tf_precision),
		# 				np.pi * tf.random_uniform([self.batch_size, self.max_num_atoms], maxval=2.0, dtype=self.tf_precision),
		# 				tf.random_uniform([self.batch_size, self.max_num_atoms], minval=0.1, maxval=1.9, dtype=self.tf_precision)], axis=-1)
		# padding_mask = tf.where(tf.not_equal(Zs_pl, 0))
		# centered_xyzs = tf.expand_dims(tf.gather_nd(xyzs_pl, padding_mask), axis=1) - tf.gather(xyzs_pl, padding_mask[:,0])
		# rotation_params = tf.gather_nd(rotation_params, padding_mask)
		# rotated_xyzs = tf_random_rotate(centered_xyzs, rotation_params)
		# embed = tf_gaush_element_channelv2(rotated_xyzs, Zs_pl, elements,
		# 							gaussian_params, self.l_max)
		# sess = tf.Session()
		# sess.run(tf.global_variables_initializer())
		# num_cases = 0
		# for ministep in range(int(0.1 * self.num_train_cases/self.batch_size)):
		# 	batch_data = self.get_train_batch(self.batch_size)
		# 	embedding = sess.run(embed, feed_dict = {xyzs_pl:batch_data[0], Zs_pl:batch_data[1]})
		# 	if ministep == 0:
		# 		self.embed_stddev = np.var(embedding, axis=0)
		# 		self.embed_mean = np.mean(embedding, axis=0)
		# 	else:
		# 		self.embed_stddev = (((self.embed_stddev * num_cases + np.var(embedding, axis=0) * embedding.shape[0])
		# 							/ (num_cases + embedding.shape[0]))
		# 							+ (np.square(self.embed_mean - np.mean(embedding, axis=0)) * num_cases * embedding.shape[0]
		# 							/ ((num_cases + embedding.shape[0]) ** 2)))
		# 		self.embed_mean = ((self.embed_mean * num_cases + np.mean(embedding, axis=0) * embedding.shape[0])
		# 						/ (num_cases + embedding.shape[0]))
		# 	num_cases += embedding.shape[0]
		# sess.close()
		# self.train_pointer = 0
		self.embed_shape = self.elements.shape[0] * self.gaussian_params.shape[0] * (self.l_max + 1) ** 2
		self.latent_shape = self.embed_shape - 3
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
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
			if self.train_sparse:
				self.pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, self.max_num_pairs, 4])

			self.gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_precision)
			elements = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
			rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size, self.max_num_atoms], maxval=2.0, dtype=self.tf_precision),
							np.pi * tf.random_uniform([self.batch_size, self.max_num_atoms], maxval=2.0, dtype=self.tf_precision),
							tf.random_uniform([self.batch_size, self.max_num_atoms], minval=0.1, maxval=1.9, dtype=self.tf_precision)], axis=-1)
			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			centered_xyzs = tf.expand_dims(tf.gather_nd(self.xyzs_pl, padding_mask), axis=1) - tf.gather(self.xyzs_pl, padding_mask[:,0])
			rotation_params = tf.gather_nd(rotation_params, padding_mask)
			rotated_xyzs = tf_random_rotate(centered_xyzs, rotation_params)
			dist_tensor = tf.norm(rotated_xyzs+1.e-16,axis=-1)
			sph_harmonics = tf.reshape(tf_spherical_harmonics(rotated_xyzs, dist_tensor, self.l_max),
						[(tf.shape(padding_mask)[0] * self.max_num_atoms), (self.l_max + 1) ** 2])
			self.invar_sph_harmonics = tf.reshape(tf_spherical_harmonics(rotated_xyzs, dist_tensor, self.l_max, invariant=True),
						[(tf.shape(padding_mask)[0] * self.max_num_atoms), (self.l_max + 1)])
			latent_vector = self.encoder(sph_harmonics)
			decoded_sph_harmonics = self.decoder(latent_vector)
			self.invar_output = tf.stack([decoded_sph_harmonics[...,0], tf.norm(decoded_sph_harmonics[...,1:4], axis=-1),
						tf.norm(decoded_sph_harmonics[...,4:9], axis=-1), tf.norm(decoded_sph_harmonics[...,9:], axis=-1)], axis=-1)
			rot_grad = tf.gradients(latent_vector, rotation_params)
			self.embed_loss = self.loss_op(self.invar_output - self.invar_sph_harmonics)
			tf.summary.scalar("embed_loss", self.embed_loss)
			tf.add_to_collection('embed_losses', self.embed_loss)
			self.rotation_loss = self.loss_op(rot_grad)
			if self.train_rotation:
				tf.add_to_collection('embed_losses', self.rotation_loss)
				tf.summary.scalar("rotation_loss", self.rotation_loss)

			self.embed_losses = tf.add_n(tf.get_collection('embed_losses'))
			tf.summary.scalar("embed_losses", self.embed_losses)

			self.train_op = self.optimizer(self.embed_losses, self.learning_rate, self.momentum)
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

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.num_train_cases
		start_time = time.time()
		train_loss =  0.0
		train_embed_loss = 0.0
		train_rotation_loss = 0.0
		num_atoms = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data = self.get_train_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_rotation:
				_, summaries, total_loss, embed_loss, rotation_loss = self.sess.run([self.train_op,
				self.summary_op, self.embed_losses, self.embed_loss, self.rotation_loss], feed_dict=feed_dict)
				train_rotation_loss += rotation_loss
			else:
				if self.profiling:
					_, summaries, total_loss, embed_loss = self.sess.run([self.train_op,
					self.summary_op, self.embed_losses, self.embed_loss], feed_dict=feed_dict,
					options=self.options, run_metadata=self.run_metadata)
					fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
					chrome_trace = fetched_timeline.generate_chrome_trace_format()
					with open('timeline_step_%d.json' % ministep, 'w') as f:
						f.write(chrome_trace)
				else:
					_, summaries, total_loss, embed_loss = self.sess.run([self.train_op,
					self.summary_op, self.embed_losses, self.embed_loss], feed_dict=feed_dict)
			train_loss += total_loss
			train_embed_loss += embed_loss
			num_atoms += np.sum(batch_data[2])
			self.summary_writer.add_summary(summaries, step * int(Ncase_train/self.batch_size) + ministep)
		duration = time.time() - start_time
		self.print_epoch(step, duration, train_loss, train_embed_loss, train_rotation_loss, num_atoms)
		return

	def test_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		Ncase_test = self.num_test_cases
		start_time = time.time()
		test_loss =  0.0
		test_embed_loss = 0.0
		test_rotation_loss = 0.0
		num_atoms = 0
		test_epoch_errors = []
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.get_test_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			total_loss, embed_loss, rotation_loss, invar_sph_harmonics, invar_output = self.sess.run([self.embed_losses, self.embed_loss,
													self.rotation_loss, self.invar_sph_harmonics, self.invar_output], feed_dict=feed_dict)
			test_loss += total_loss
			test_embed_loss += embed_loss
			test_rotation_loss += rotation_loss
			test_epoch_errors.append(invar_output - invar_sph_harmonics)
			num_atoms += np.sum(batch_data[2]) * self.max_num_atoms
		test_epoch_errors = np.concatenate(test_epoch_errors)
		test_mse = np.mean(test_epoch_errors)
		test_mae = np.mean(np.abs(test_epoch_errors))
		test_rmse = np.sqrt(np.mean(np.square(test_epoch_errors)))
		LOGGER.info("MAE : %11.8f", test_mae)
		LOGGER.info("MSE : %11.8f", test_mse)
		LOGGER.info("RMSE: %11.8f", test_rmse)
		duration = time.time() - start_time
		self.print_epoch(step, duration, test_loss, test_embed_loss, test_rotation_loss, num_atoms, testing=True)
		return test_loss

class GauSHEncoderv2(GauSHEncoder):
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
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
			if self.train_sparse:
				self.pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, self.max_num_pairs, 4])

			self.gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_precision)
			elements = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
			rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size, self.max_num_atoms], maxval=2.0, dtype=self.tf_precision),
							np.pi * tf.random_uniform([self.batch_size, self.max_num_atoms], maxval=2.0, dtype=self.tf_precision),
							tf.random_uniform([self.batch_size, self.max_num_atoms], minval=0.1, maxval=1.9, dtype=self.tf_precision)], axis=-1)
			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			centered_xyzs = tf.expand_dims(tf.gather_nd(self.xyzs_pl, padding_mask), axis=1) - tf.gather(self.xyzs_pl, padding_mask[:,0])
			dist_tensor = tf.norm(centered_xyzs+1.e-16,axis=-1)
			embed = tf_gaush_element_channelv2(centered_xyzs, self.Zs_pl, elements, self.gaussian_params, self.l_max)
			rotation_params = tf.gather_nd(rotation_params, padding_mask)
			rotated_xyzs = tf_random_rotate(centered_xyzs, rotation_params)
			dist_tensor = tf.norm(rotated_xyzs+1.e-16,axis=-1)
			self.rot_embed = tf_gaush_element_channelv2(rotated_xyzs, self.Zs_pl, elements, self.gaussian_params, self.l_max)
			latent_vector = self.encoder(embed)
			latent_embed = latent_vector[...,:-3]
			latent_angles = latent_vector[...,-3:]
			latent_shift_angles = latent_angles + rotation_params
			shift_latent_vector = tf.concat([latent_embed, latent_shift_angles], axis=1)
			self.decoded_embed = self.decoder(shift_latent_vector)
			# rot_grad = tf.gradients(latent_embed, rotation_params)
			self.embed_loss = self.loss_op(self.decoded_embed - self.rot_embed)
			tf.summary.scalar("embed_loss", self.embed_loss)
			tf.add_to_collection('embed_losses', self.embed_loss)
			# self.rotation_loss = self.loss_op(rot_grad)
			# if self.train_rotation:
			# 	tf.add_to_collection('embed_losses', self.rotation_loss)
			# 	tf.summary.scalar("rotation_loss", self.rotation_loss)

			self.embed_losses = tf.add_n(tf.get_collection('embed_losses'))
			tf.summary.scalar("embed_losses", self.embed_losses)

			self.train_op = self.optimizer(self.embed_losses, self.learning_rate, self.momentum)
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

	def encoder(self, inputs):
		"""
		Builds an encoder for the GauSH descriptor

		Args:
			inputs (tf.float): NCase x Embed Shape of the embeddings to encode
		Returns:
			The latent space vectors
		"""
		with tf.name_scope("encoder"):
			if len(self.hidden_layers) == 0:
				with tf.name_scope('encoder_hidden'):
					weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.embed_shape],
							stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([self.embed_shape], dtype=self.tf_precision), name='biases')
					latent_vectors = self.activation_function(tf.matmul(inputs, weights) + biases)
			else:
				for i in range(len(self.hidden_layers)):
					if i == 0:
						with tf.name_scope('encoder_hidden1'):
							weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[i]],
									stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
							hidden_output = self.activation_function(tf.matmul(inputs, weights) + biases)
					else:
						with tf.name_scope('encoder_hidden'+str(i+1)):
							weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
									stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
							hidden_output = self.activation_function(tf.matmul(hidden_output, weights) + biases)
				with tf.name_scope('latent_space'):
					weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], self.embed_shape],
							stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([self.embed_shape], dtype=self.tf_precision), name='biases')
					latent_vectors = tf.matmul(hidden_output, weights) + biases
		return latent_vectors

	def decoder(self, latent_vector):
		"""
		Builds a decoder for the GauSH descriptor

		Args:
			inputs (tf.float): NCase x latent vector shape tensor of the embedding cases
		Returns:
			The decoded GauSH descriptor
		"""
		with tf.name_scope("decoder"):
			if len(self.hidden_layers) == 0:
				with tf.name_scope('encoder_hidden'):
					weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.embed_shape],
							stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([self.embed_shape], dtype=self.tf_precision), name='biases')
					outputs = self.activation_function(tf.matmul(latent_vector, weights) + biases)
			else:
				for i in range(len(self.hidden_layers)):
					if i == 0:
						with tf.name_scope('decoder_hidden1'):
							weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[-1]],
									stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[-1]], dtype=self.tf_precision), name='biases')
							hidden_output = self.activation_function(tf.matmul(latent_vector, weights) + biases)
					else:
						with tf.name_scope('decoder_hidden'+str(i+1)):
							weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-i], self.hidden_layers[-1-i]],
									stddev=math.sqrt(2.0 / float(self.hidden_layers[-i])), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[-1-i]], dtype=self.tf_precision), name='biases')
							hidden_output = self.activation_function(tf.matmul(hidden_output, weights) + biases)
				with tf.name_scope('decoder_output'):
					weights = self.variable_with_weight_decay(shape=[self.hidden_layers[0], self.embed_shape],
							stddev=math.sqrt(2.0 / float(self.hidden_layers[0])), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([self.embed_shape], dtype=self.tf_precision), name='biases')
					outputs = tf.matmul(hidden_output, weights) + biases
		return outputs

	def test_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		Ncase_test = self.num_test_cases
		start_time = time.time()
		test_loss =  0.0
		test_embed_loss = 0.0
		test_rotation_loss = 0.0
		num_atoms = 0
		test_epoch_errors = []
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.get_test_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			total_loss, embed_loss, rot_embed, decoded_embed = self.sess.run([self.embed_losses, self.embed_loss,
						self.rot_embed, self.decoded_embed], feed_dict=feed_dict)
			test_loss += total_loss
			test_embed_loss += embed_loss
			# test_rotation_loss += rotation_loss
			test_epoch_errors.append(rot_embed - decoded_embed)
			num_atoms += np.sum(batch_data[2]) * self.max_num_atoms
		test_epoch_errors = np.concatenate(test_epoch_errors)
		test_mse = np.mean(test_epoch_errors)
		test_mae = np.mean(np.abs(test_epoch_errors))
		test_rmse = np.sqrt(np.mean(np.square(test_epoch_errors)))
		LOGGER.info("MAE : %11.8f", test_mae)
		LOGGER.info("MSE : %11.8f", test_mse)
		LOGGER.info("RMSE: %11.8f", test_rmse)
		duration = time.time() - start_time
		self.print_epoch(step, duration, test_loss, test_embed_loss, test_rotation_loss, num_atoms, testing=True)
		return test_loss

class GauSHTranscoder(GauSHEncoder):
	"""
	Class for training a transforming autoencoder on the
	spherical harmonics.
	"""
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
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
			if self.train_sparse:
				self.pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, self.max_num_pairs, 4])

			self.gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_precision)
			elements = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
			rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size, self.max_num_atoms], maxval=2.0, dtype=self.tf_precision),
							np.pi * tf.random_uniform([self.batch_size, self.max_num_atoms], maxval=2.0, dtype=self.tf_precision),
							tf.random_uniform([self.batch_size, self.max_num_atoms], minval=0.1, maxval=1.9, dtype=self.tf_precision)], axis=-1)
			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			centered_xyzs = tf.expand_dims(tf.gather_nd(self.xyzs_pl, padding_mask), axis=1) - tf.gather(self.xyzs_pl, padding_mask[:,0])
			rotation_params = tf.gather_nd(rotation_params, padding_mask)
			rotated_xyzs = tf_random_rotate(centered_xyzs, rotation_params)
			dist_tensor = tf.norm(rotated_xyzs+1.e-16,axis=-1)
			sph_harmonics = tf.reshape(tf_spherical_harmonics(rotated_xyzs, dist_tensor, self.l_max),
						[(tf.shape(padding_mask)[0] * self.max_num_atoms), (self.l_max + 1) ** 2])
			self.invar_sph_harmonics = tf.reshape(tf_spherical_harmonics(rotated_xyzs, dist_tensor, self.l_max, invariant=True),
						[(tf.shape(padding_mask)[0] * self.max_num_atoms), (self.l_max + 1)])
			latent_vector = self.encoder(sph_harmonics)
			latent_sh = latent_vector[...,:-3]
			latent_rot_params = latent_vector[...,-3:]
			decoded_sph_harmonics = self.decoder(latent_vector)
			self.invar_output = tf.stack([decoded_sph_harmonics[...,0], tf.norm(decoded_sph_harmonics[...,1:4], axis=-1),
						tf.norm(decoded_sph_harmonics[...,4:9], axis=-1), tf.norm(decoded_sph_harmonics[...,9:], axis=-1)], axis=-1)
			rot_grad = tf.gradients(latent_vector, rotation_params)

			self.embed_loss = self.loss_op(self.invar_output - self.invar_sph_harmonics)
			tf.summary.scalar("embed_loss", self.embed_loss)
			tf.add_to_collection('embed_losses', self.embed_loss)
			self.rotation_loss = self.loss_op(rot_grad)
			if self.train_rotation:
				tf.add_to_collection('embed_losses', self.rotation_loss)
				tf.summary.scalar("rotation_loss", self.rotation_loss)

			self.embed_losses = tf.add_n(tf.get_collection('embed_losses'))
			tf.summary.scalar("embed_losses", self.embed_losses)

			self.train_op = self.optimizer(self.embed_losses, self.learning_rate, self.momentum)
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

	def capsule(self, input, extra_input):
		with tf.name_scope('recognizer'):
			weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[0]],
					stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
			biases = tf.Variable(tf.zeros([self.hidden_layers[0]], dtype=self.tf_precision), name='biases')
			recognition = self.activation_function(tf.matmul(inputs, weights) + biases)
		with tf.name_scope('probability'):
			weights = self.variable_with_weight_decay(shape=[self.hidden_layers[0], 1],
					stddev=math.sqrt(2.0 / float(self.hidden_layers[0])), weight_decay=self.weight_decay, name="weights")
			biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
			probability = self.activation_function(tf.matmul(recognition, weights) + biases)
		with tf.name_scope('transformation'):
			weights = self.variable_with_weight_decay(shape=[self.hidden_layers[0], 9],
					stddev=math.sqrt(2.0 / float(self.hidden_layers[0])), weight_decay=self.weight_decay, name="weights")
			biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
			learned_rotation = tf.matmul(recognition, weights) + biases
			dlearned_rotation = tf.matmul(learned_rotation, extra_input)
		with tf.name_scope('generator'):
			weights = self.variable_with_weight_decay(shape=[9, self.hidden_layers[1]],
					stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
			biases = tf.Variable(tf.zeros([self.hidden_layers[0]], dtype=self.tf_precision), name='biases')
			generation = self.activation_function(tf.matmul(dlearned_rotation, weights) + biases)
		with tf.name_scope('output'):
			weights = self.variable_with_weight_decay(shape=[9, self.hidden_layers[1]],
					stddev=math.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
			biases = tf.Variable(tf.zeros([self.hidden_layers[0]], dtype=self.tf_precision), name='biases')
			output = tf.matmul(generation, weights) + biases
			output *= probability
		return output

	def encoder(self, input, extra_input):
		"""
		Builds an encoder for the GauSH descriptor

		Args:
			inputs (tf.float): NCase x Embed Shape of the embeddings to encode
		Returns:
			The latent space vectors
		"""
		for i in range(self.num_capsules):
			with tf.name_scope('capsule_'+str(i)):
				self.capsules_outputs.append(self.capsule(inputs, extra_input))
		reconstruct = tf.add_n(self.capsule_outputs)
		return reconstruct
