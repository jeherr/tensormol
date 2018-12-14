from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..ForceModifiers.Neighbors import *
from ..Math.TFMath import * # Why is this imported here?
from ..Math.LinearOperations import *
from ..ElementData import *
from ..TFDescriptors.RawSH import *
from tensorflow.python.client import timeline
if (HAS_TF):
	import tensorflow as tf

class ElementCoder(object):
	"""
	Base Class for a Kingma-style variational autoencoder
	with a few bells and whistles (Gradient Clipping)

	TODO: Choose intelligent Initializers.
	"""
	def __init__(self, latent_size=4, batches_per_epoch=100):
		self.tf_precision = eval(PARAMS["tf_prec"])
		self.hidden_layers = PARAMS["HiddenLayers"]
		self.learning_rate = PARAMS["learning_rate"]
		self.weight_decay = PARAMS["weight_decay"]
		self.momentum = PARAMS["momentum"]
		self.max_steps = PARAMS["max_steps"]
		self.batch_size = PARAMS["batch_size"]
		self.max_checkpoints = PARAMS["max_checkpoints"]
		self.path = PARAMS["networks_directory"]
		self.activation_function_type = PARAMS["NeuronType"]
		self.assign_activation()
		self.step = 0
		self.test_freq = PARAMS["test_freq"]
		self.network_type = "E_Coder"
		self.name = self.network_type+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.network_directory = PARAMS["networks_directory"]+self.name
		self.latent_size = latent_size
		self.batches_per_epoch = batches_per_epoch

		self.atom_data = AtomData
		self.atom_features = np.array([data[2:] for data in AtomData], dtype=np.float64)
		self.feature_length = np.shape(self.atom_features)[1]
		self.data_mean = np.mean(self.atom_features, axis=0)
		self.data_std = np.std(self.atom_features, axis=0)

		LOGGER.info("learning rate: %f", self.learning_rate)
		LOGGER.info("batch size:    %d", self.batch_size)
		LOGGER.info("max steps:     %d", self.max_steps)
		return

	def assign_activation(self):
		LOGGER.debug("Assigning Activation Function: %s", PARAMS["NeuronType"])
		try:
			if self.activation_function_type == "relu":
				self.activation_function = tf.nn.relu
			elif self.activation_function_type == "elu":
				self.activation_function = tf.nn.elu
			elif self.activation_function_type == "selu":
				self.activation_function = tf.nn.selu
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

	def train(self):
		self.build_graph()
		for i in range(self.max_steps):
			self.step += 1
			self.train_step()
			if self.step % self.test_freq == 0:
				test_loss = self.test_step()
				if self.step == self.test_freq:
					self.best_loss = test_loss
					self.save_checkpoint()
				elif test_loss < self.best_loss:
					self.best_loss = test_loss
					self.save_checkpoint()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		batch_data = self.atom_features[1:,0]
		feed_dict = self.fill_feed_dict(batch_data)
		latent_features = self.sess.run(self.latent_features,  feed_dict=feed_dict)
		latent_shape = latent_features.shape
		strng = "["
		for i in range(latent_shape[0]):
			strng += "["
			for j in range(latent_shape[1]):
				strng += str(latent_features[i,j])
				if j != (latent_shape[1]-1):
					strng += ", "
			if i != (latent_shape[0]-1):
				strng += "],\n"
			else:
				strng += "]"
		strng += "]"
		print(strng)
		self.sess.close()
		return

	def build_graph(self, restart=False):
		self.Zs_pl = tf.placeholder(tf.int32, shape=[None])
		self.tf_atom_features = tf.Variable(self.atom_features, trainable=False, dtype = self.tf_precision)

		self.gather_idx = tf.where(tf.equal(tf.expand_dims(tf.cast(self.Zs_pl, self.tf_precision), axis=-1),
				self.tf_atom_features[:,0]))[:,1]
		self.batch_features = tf.gather(self.tf_atom_features, self.gather_idx)
		self.norm_batch_features = (self.batch_features) / self.data_std
		self.latent_features = self.encoder(self.batch_features)
		self.norm_decoded_features = self.decoder(self.latent_features)
		self.decoded_features = (self.norm_decoded_features * self.data_std)
		self.reconstruction_loss = self.loss_op(self.norm_batch_features - self.norm_decoded_features)
		self.train_op = self.optimizer(self.reconstruction_loss, self.learning_rate, self.momentum)
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
		if restart:
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)
		return

	def encoder(self, features):
		for i in range(len(self.hidden_layers)):
			if i == 0:
				layer = tf.layers.dense(inputs=features, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)
			else:
				layer = tf.layers.dense(inputs=layer, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)

		latent_features = tf.layers.dense(inputs=layer, units=self.latent_size,
				activation=self.activation_function, use_bias=True)
		return latent_features

	def decoder(self, latent_features):
		for i in range(len(self.hidden_layers)):
			if i == 0:
				layer = tf.layers.dense(inputs=latent_features, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)
			else:
				layer = tf.layers.dense(inputs=layer, units=self.hidden_layers[i],
						activation=self.activation_function, use_bias=True)

		decoded_features = tf.layers.dense(inputs=layer, units=self.feature_length,
				activation=None, use_bias=True)
		return decoded_features

	def optimizer(self, loss, learning_rate, momentum):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def loss_op(self, error):
		loss = tf.nn.l2_loss(error)
		return loss

	def train_step(self):
		start_time = time.time()
		train_loss =  0.0
		for ministep in range(self.batches_per_epoch):
			batch_data = np.random.choice(self.atom_features[1:,0], size=self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			_, loss = self.sess.run([self.train_op, self.reconstruction_loss], feed_dict=feed_dict)
			train_loss += loss
		train_loss /= self.batches_per_epoch
		train_loss /= self.batch_size
		duration = time.time() - start_time
		print("step:", self.step, " duration:", duration, " reconstruction loss:", train_loss)
		return

	def test_step(self):
		print("testing...")
		start_time = time.time()
		test_loss =  0.0
		batch_data = self.atom_features[1:,0]
		feed_dict = self.fill_feed_dict(batch_data)
		test_loss, decoded_features, batch_features = self.sess.run([self.reconstruction_loss, self.decoded_features, self.batch_features],  feed_dict=feed_dict)
		test_loss /= np.shape(batch_data)
		duration = time.time() - start_time
		print((self.atom_features[1:] - decoded_features))
		# for i in np.random.choice(np.arange(1,self.atom_features.shape[0]-1), size=10):
		# 	print("   Atom Features:", self.atom_features[i+1])
		# 	print("Decoded Features:", decoded_features[i])
		# print("MAE  Energy:", np.mean(np.abs(test_energy_errors)))
		# print("MSE  Energy:", np.mean(test_energy_errors))
		# print("RMSE Energy:", np.sqrt(np.mean(np.square(test_energy_errors))))
		print("step:", self.step, " duration:", duration, " reconstruction loss:", test_loss)
		return test_loss

	def fill_feed_dict(self, batch_data):
		feed_dict={self.Zs_pl:batch_data}
		return feed_dict

	def save_checkpoint(self):
		checkpoint_file = os.path.join(self.network_directory,self.name+'-checkpoint')
		self.saver.save(self.sess, checkpoint_file, global_step=self.step)
		return
