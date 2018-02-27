"""
- A BP which uses encoder embeddings rather than the symmetry functions directly.
 This is also written without the inheritance cruft of the older instances, although the interface is
 kept the same.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import sys
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle
#
# A Variation on BP which uses a classification channel.
#
class BPCoder(object):
	'''
	I'm so sorry guys.
	I'm trying really hard to make this class work.
	<3 -Ryker
	'''
	def __init__(self):

		# Network params
		self.learning_rate = PARAMS["learning_rate"]
		self.max_steps = PARAMS["max_steps"]
		self.batch_size = PARAMS["batch_size"]

		# Training params
		self.hidden_layers = PARAMS["HiddenLayers"]

		# Grab num_input from somewhere?? --- NEEDS FIXED
		self.num_input = 1000

		# Graph input
		self.X = tf.placeholder("float", [None, num_input])

		# Build weights and biases dicts
		self.energy_inference(self, SOMETHING, WHO KNOWS)

		# Run
		self.run(self)

		return

	def __getstate__(self):
		state = self.__dict__.copy()
		remove_vars = []
		for var in remove_vars:
			try:
				del state[var]
			except:
				pass
		return state

	def run(self):
		# Construct model
		self.encoder_op = encoder(self.X)
		self.decoder_op = decoder(self.encoder_op)

		# Prediction
		self.y_pred = self.decoder_op
		# Targets (Labels) are the input data.
		self.y_true = self.X

		# Define loss and optimizer, minimize the squared error
		self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

		# Initialize the variables (i.e. assign their default value)
		self.init = tf.global_variables_initializer()

		return

	def energy_inference(self, inp, indexs):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		branches=[]
		variables=[]
		output = tf.zeros([self.batch_size, self.max_num_atoms], dtype=self.tf_precision)
		with tf.name_scope("energy_network"):
			for e in range(len(self.elements)):
				branches.append([])
				inputs = inp[e]
				index = indexs[e]
				for i in range(len(self.hidden_layers)):
					if i == 0:
						with tf.name_scope(str(self.elements[e])+'_hidden1'):
							weights = self.variable_with_weight_decay(shape=[self.embedding_shape, self.hidden_layers[i]],
									stddev=math.sqrt(2.0 / float(self.embedding_shape)), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
							branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
							variables.append(weights)
							variables.append(biases)
					else:
						with tf.name_scope(str(self.elements[e])+'_hidden'+str(i+1)):
							weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
									stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
							biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
							branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
							variables.append(weights)
							variables.append(biases)
				with tf.name_scope(str(self.elements[e])+'_regression_linear'):
					weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
							stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
					branches[-1].append(tf.squeeze(tf.matmul(branches[-1][-1], weights) + biases, axis=1))
					variables.append(weights)
					variables.append(biases)
					output += tf.scatter_nd(index, branches[-1][-1], [self.batch_size, self.max_num_atoms])
				tf.verify_tensor_all_finite(output,"Nan in output!!!")
		return tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size]), variables

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

	def encoder(self, x):
		# Encoder Hidden layer with sigmoid activation #1
		self.layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
		# Encoder Hidden layer with sigmoid activation #2
		self.layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(self.layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
		return self.layer_2

	def decoder(self, x):
		# Decoder Hidden layer with sigmoid activation #1
		self.layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
		# Decoder Hidden layer with sigmoid activation #2
		self.layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(self.layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
		return self.layer_2

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

	def reload_set(self):
		"""
		Recalls the MSet to build training data etc.
		"""
		self.mol_set = MSet(self.mol_set_name)
		self.mol_set.Load()
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

	def compute_normalization(self):
		elements = tf.constant(self.elements, dtype = tf.int32)
		element_pairs = tf.constant(self.element_pairs, dtype = tf.int32)
		radial_rs = tf.constant(self.radial_rs, dtype = self.tf_precision)
		angular_rs = tf.constant(self.angular_rs, dtype = self.tf_precision)
		theta_s = tf.constant(self.theta_s, dtype = self.tf_precision)
		radial_cutoff = tf.constant(self.radial_cutoff, dtype = self.tf_precision)
		angular_cutoff = tf.constant(self.angular_cutoff, dtype = self.tf_precision)
		zeta = tf.constant(self.zeta, dtype = self.tf_precision)
		eta = tf.constant(self.eta, dtype = self.tf_precision)
		xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
		Zs_pl = tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.max_num_atoms]))
		embeddings, molecule_indices = tf_symmetry_functions(xyzs_pl, Zs_pl, elements, element_pairs, radial_cutoff,
										angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)
		embeddings_list = [[], [], [], []]
		labels_list = []

		self.embeddings_max = []
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		for ministep in range (0, max(2, int(0.1 * self.num_train_cases/self.batch_size))):
			batch_data = self.get_energy_train_batch(self.batch_size)
			labels_list.append(batch_data[2])
			embedding, molecule_index = sess.run([embeddings, molecule_indices], feed_dict = {xyzs_pl:batch_data[0], Zs_pl:batch_data[1]})
			for element in range(len(self.elements)):
				embeddings_list[element].append(embedding[element])
		sess.close()
		for element in range(len(self.elements)):
			self.embeddings_max.append(np.amax(np.concatenate(embeddings_list[element])))
		labels = np.concatenate(labels_list)
		self.labels_mean = np.mean(labels)
		self.labels_stddev = np.std(labels)
		self.train_pointer = 0

		#Set the embedding and label shape
		self.embedding_shape = embedding[0].shape[1]
		self.label_shape = labels[0].shape
		return

	def save_network(self):
		print("Saving TFInstance")
		f = open(self.network_directory+".tfn","wb")
		pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

	def train_prepare(self,  continue_training =False):
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
			self.dipole_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, 3])
			self.quadrupole_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, 2, 3])
			self.gradients_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
			self.Reep_pl = tf.placeholder(tf.int32, shape=[None, 3])

			self.dipole_labels = self.dipole_pl
			self.quadrupole_labels = self.quadrupole_pl

			elements = tf.constant(self.elements, dtype = tf.int32)
			element_pairs = tf.constant(self.element_pairs, dtype = tf.int32)
			radial_rs = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_rs = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			theta_s = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			elu_width = tf.Variable(self.elu_width * BOHRPERA, trainable=False, dtype = self.tf_precision)
			dsf_alpha = tf.Variable(self.dsf_alpha, trainable=False, dtype = self.tf_precision)
			coulomb_cutoff = tf.Variable(self.coulomb_cutoff, trainable=False, dtype = self.tf_precision)

			embeddings_max = tf.constant(self.embeddings_max, dtype = self.tf_precision)
			labels_mean = tf.constant(self.labels_mean, dtype = self.tf_precision)
			labels_stddev = tf.constant(self.labels_stddev, dtype = self.tf_precision)
			num_atoms_batch = tf.reduce_sum(self.num_atoms_pl)

			embeddings, mol_idx = tf_symmetry_functions(self.xyzs_pl, self.Zs_pl, elements,
					element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)
			for element in range(len(self.elements)):
				embeddings[element] /= embeddings_max[element]
			norm_bp_energy, energy_variables = self.energy_inference(embeddings, mol_idx)
			self.bp_energy = (norm_bp_energy * self.labels_stddev) + self.labels_mean

			if self.train_dipole:
				self.dipoles, self.quadrupoles, self.charges, self.net_charge, dipole_variables = self.dipole_inference(embeddings, mol_idx, self.xyzs_pl, self.num_atoms_pl)
				if (PARAMS["OPR12"]=="DSF"):
					self.coulomb_energy = tf_coulomb_dsf_elu(rotated_xyzs, self.charges, self.Reep_pl, elu_width, dsf_alpha, coulomb_cutoff)
				elif (PARAMS["OPR12"]=="Poly"):
					self.coulomb_energy = PolynomialRangeSepCoulomb(self.xyzs_pl, self.charges, self.Reep_pl, 5.0, 12.0, 5.0)
				self.total_energy = self.bp_energy + self.coulomb_energy
				self.dipole_loss = self.loss_op(self.dipoles - self.dipole_pl)
				self.quadrupole_loss = self.loss_op(self.quadrupoles - self.quadrupole_pl)
				tf.add_to_collection('dipole_losses', self.dipole_loss)
				tf.add_to_collection('quadrupole_losses', self.quadrupole_loss)
				self.dipole_losses = tf.add_n(tf.get_collection('dipole_losses'))
				tf.summary.scalar("dipole losses", self.dipole_losses)
				self.dipole_train_op = self.optimizer(self.dipole_losses, self.learning_rate, self.momentum, dipole_variables)
			else:
				self.total_energy = self.bp_energy

			self.gradients = tf.gather_nd(tf.gradients(self.bp_energy, self.xyzs_pl)[0], tf.where(tf.not_equal(self.Zs_pl, 0)))
			self.gradient_labels = tf.gather_nd(self.gradients_pl, tf.where(tf.not_equal(self.Zs_pl, 0)))
			self.energy_loss = self.loss_op(self.total_energy - self.energy_pl)
			tf.summary.scalar("energy loss", self.energy_loss)
			tf.add_to_collection('energy_losses', self.energy_loss)
			self.gradient_loss = self.loss_op(self.gradients - self.gradient_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
			if self.train_gradients:
				tf.add_to_collection('energy_losses', self.gradient_loss)
				tf.summary.scalar("gradient loss", self.gradient_loss)

			self.energy_losses = tf.add_n(tf.get_collection('energy_losses'))
			tf.summary.scalar("energy losses", self.energy_losses)

			self.energy_train_op = self.optimizer(self.energy_losses, self.learning_rate, self.momentum, energy_variables)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
			self.sess.run(init)
			if self.profiling:
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
		return

	def train(self):
		test_freq = PARAMS["test_freq"]
		if self.train_dipole:
			mini_test_loss = 1e10
			for step in range(1, 51):
				self.dipole_train_step(step)
				if step%test_freq==0:
					test_loss = self.dipole_test_step(step)
					if (test_loss < mini_test_loss):
						mini_test_loss = test_loss
						self.save_checkpoint(step)
			LOGGER.info("Continue training dipole until new best checkpoint found.")
			train_energy_flag = False
			step += 1
			while train_energy_flag == False:
				self.dipole_train_step(step)
				test_loss = self.dipole_test_step(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_checkpoint(step)
					train_energy_flag=True
					LOGGER.info("New best checkpoint found. Starting energy network training.")
				step += 1
		mini_test_loss = 1e10
		for step in range(1, self.max_steps+1):
			self.energy_train_step(step)
			if step%test_freq==0:
				test_loss = self.energy_test_step(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_checkpoint(step)
		self.sess.close()
		self.save_network()
		return
