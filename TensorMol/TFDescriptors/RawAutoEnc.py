"""
Unlike a symmetry function, an encoder embedding doesn't encode atomic numbers into channels
Ie: it doesn't have pair or triples element channels. Instead it maps every atom onto some basis of
vectors unsupervised channels which arise, for example from an autoencoder. These channels are then
used in a traditional geometric descriptor (GauSH or SF).
They can also be used to fuse atoms together, finding classes for substructures.
They must be trained alongside a neural network model.

The plus to this approach is you avoid bond pairs, triples insanity with element types.

Essential operation is:
ZXYZ => (Short vectors) => ZXYZ (up to invariance)
But there are different cases:

-Embedding encoder
  This encoder reproduces some information about nearby atoms
  in the vector for each atom. However each atom gets its own
  embedding.

-Compressive encoder
  This encoder maps many atoms onto possibly few outputs as invertibly as possible.

But the short vectors could be asked to reproduce information about nearby atoms.
For example can we get a 4-10 dimensional embedding of an atoms hybridization or charge.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..ForceModifiers.Neighbors import *
from ..Math.TFMath import * # Why is this imported here?
from ..Math.LinearOperations import *
from ..ElementData import *
from .RawSH import *
from tensorflow.python.client import timeline
if (HAS_TF):
	import tensorflow as tf

class VariationalAutoencoder(object):
	"""
	Base Class for a Kingma-style variational autoencoder
	with a few bells and whistles (Gradient Clipping)

	TODO: Choose intelligent Initializers.
	"""
	def __init__(self):
		self.batch_size = 100
		self.learning_rate = 0.002
		self.n_latent = 5
		self.act_fcn = tf.nn.selu
		return
	def Encoder(self,in_):
		l1 = tf.layers.dense(inputs=in_, units=64, activation=act_fcn, use_bias=True)
		l2 = tf.layers.dense(inputs=l1, units=64, activation=act_fcn, use_bias=True)
		latent_vector = tf.layers.dense(inputs=l2, units=self.n_latent, activation=act_fcn, use_bias=True)
		z_mean = tf.layers.dense(inputs=latent_vector, units=self.n_latent, activation=None, use_bias=True)
		z_log_sigma = tf.layers.dense(inputs=latent_vector, units=self.n_latent, activation=None, use_bias=True)
		return latent_vector, z_mean, z_log_sigma
	def Decoder(self,in_,target_shape_):
		l1 = tf.layers.dense(inputs=in_, units=64, activation=act_fcn, use_bias=True)
		l2 = tf.layers.dense(inputs=l1, units=64, activation=act_fcn, use_bias=True)
		l3 = tf.layers.dense(inputs=l2, units=target_shape_, activation=act_fcn, use_bias=True)
		return
	@staticmethod
	def sampleGaussian(mu, log_sigma):
		"""
		The noise process which regularizes the autoencoder
		"""
		with tf.name_scope("sample_gaussian"):
			# reparameterization trick
			epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
			return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)
	@staticmethod
	def crossEntropy(obs, actual, offset=1e-7):
		"""Binary cross-entropy, per training example"""
		# (tf.Tensor, tf.Tensor, float) -> tf.Tensor
		with tf.name_scope("cross_entropy"):
			# bound by clipping to avoid nan
			obs_ = tf.clip_by_value(obs, offset, 1 - offset)
			return -tf.reduce_sum(actual * tf.log(obs_) + (1 - actual) * tf.log(1 - obs_), 1)
	@staticmethod
	def l1_loss(obs, actual):
		"""L1 loss (a.k.a. LAD), per training example"""
		# (tf.Tensor, tf.Tensor, float) -> tf.Tensor
		with tf.name_scope("l1_loss"):
			return tf.reduce_sum(tf.abs(obs - actual) , 1)
	@staticmethod
	def crossEntropy(obs, actual, offset=1e-15):
		"""Binary cross-entropy, per training example"""
		# (tf.Tensor, tf.Tensor, float) -> tf.Tensor
		with tf.name_scope("cross_entropy"):
			obs_ = tf.clip_by_value(obs, offset, 1 - offset)# bound by clipping to avoid nan
			return -tf.reduce_sum(actual * tf.log(obs_) + (1 - actual) * tf.log(1 - obs_), 1)
	@staticmethod
	def kullbackLeibler(mu, log_sigma):
		"""(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
		# (tf.Tensor, tf.Tensor) -> tf.Tensor
		with tf.name_scope("KL_divergence"):
			# = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
			return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 -tf.exp(2 * log_sigma), 1)
	def composeAE(in_):
		"""
		Build a non-bayesian Autoencoder. This one might allow collapse.
		"""
		return
	def composeVAE(in_):
		"""
		Put together the autoencoder, encoder is a tensor which depends on the input.
		"""
		with tf.name_scope("VAE"):
			encoded, z_mean, z_log_sigma = self.Encoder(in_)
			z = VariationalAutoencoder.sampleGaussian(z_mean,z_log_sigma)
			decoded = self.Decoder(z, tf.shape(in_)[1:])

		with tf.name_scope("cost"):
			# rec_loss =
			# average over minibatch
			cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
			cost += l2_reg

		with tf.name_scope("Adam_optimizer"):
			global_step = tf.Variable(0, trainable=False)
			optimizer = tf.train.AdamOptimizer(self.learning_rate)
			tvars = tf.trainable_variables()
			grads_and_vars = optimizer.compute_gradients(cost, tvars)
			clipped = [(tf.clip_by_value(grad, -5, 5), tvar) for grad, tvar in grads_and_vars]
			train_op = optimizer.apply_gradients(clipped, global_step=global_step,
				name="minimize_cost")
			return (encoded,decoded,loss,global_step)

class CapsuleNetwork(VariationalAutoencoder):
	"""
		A transforming autoencoder.
		This one learns a decoder which can rotate, and a
		probabilistic encoding of its input which should be
		rotationally invariant.

		Args:
			xyzs_ : NMol X MaxNAtom X 3 tensor of coordinates.
	"""
	def __init__(self):
		VariationalAutoencoder.__init__(self)
		self.lmax = 4
		self.MaxNAtom = 7
		self.n_capsules = 42
		self.t_dim = 3 # Dimension of Transformation
		self.r_dim = 20 # Recognizer dimension.
		self.g_dim = 10 # Generator dimension.
		self.n_rot = 40 # Number of rotations
		self.sess = None
		self.Prepare()
		return
	def Capsules(self,in_,ts_, Decode = True):
		"""
		A Hintonian Capsule for a 3-d transformation.

		Args:
			in_: the input to be reconstructed.
			ts_: transformation input (extra_input)
		"""
		act_fcn = tf.nn.relu
		probs = []
		outs = []
		for i in range(self.n_capsules):
			R1 = tf.layers.dense(inputs=in_, units=128, activation=act_fcn, use_bias=True)
			R2 = tf.layers.dense(inputs=R1, units=128, activation=act_fcn, use_bias=True)
			Prob = tf.layers.dense(inputs=R2, units=1, activation=tf.nn.relu, use_bias=True)
			T1 = tf.layers.dense(inputs=R2, units=128, activation=act_fcn, use_bias=True)
			T2 = tf.layers.dense(inputs=T1, units=self.t_dim, activation=None, use_bias=True)
			Gin = T2 + ts_
			G1 = tf.layers.dense(inputs=Gin, units=128, activation=act_fcn, use_bias=True)
			G2 = tf.layers.dense(inputs=G1, units=128, activation=act_fcn, use_bias=True)
			Out = tf.layers.dense(inputs=G2, units=(self.lmax+1)**2, activation=None, use_bias=True)
			probs.append(Prob)
			outs.append(Out)
		return tf.stack(probs),tf.add_n(outs)
	def GetBatch(self,max_dist = 0.0):
		xyzs = np.random.random(size=(self.batch_size,self.MaxNAtom,3))*7.0
		return xyzs
	def Train(self):
		step=0
		while(True):
			xyzs  = self.GetBatch()
			# Check that the rotations are a reversible transformation.
			feed_dict = {self.xyzs:xyzs,self.frac_sphere:min(1.0,step/10000.)}
			_, train_loss, step, dLdR = self.sess.run([self.train_op, self.loss, self.global_step, self.dLdR], feed_dict=feed_dict)
			print("Step: ", step, " ", train_loss)
			print("XXX", dLdR)
			if (step%100==0):
				xyzs = self.GetBatch()
				feed_dict = {self.xyzs:xyzs,self.frac_sphere:min(1.0,step/10000.)}
				emb,real,cap = self.sess.run([self.Embedded, self.Embedded_t, self.Output], feed_dict=feed_dict)
				print("emb 0 ", emb[0])
				print("Real 0 ", real[0])
				print("Cap 0 ", cap[0])
	def Prepare(self):
		"""
		Build the required graph.
		Also build evaluations.
		"""
		with tf.name_scope("Capsules"):
			self.xyzs = tf.placeholder(dtype=tf.float64, shape=(self.batch_size,self.MaxNAtom,3))
			self.frac_sphere = tf.placeholder(dtype=tf.float64)

			thetas = tf.acos(2.0*tf.random_uniform([self.batch_size],dtype=tf.float64)-1)
			phis = tf.random_uniform([self.batch_size],dtype=tf.float64)*2*Pi
			psis = tf.random_uniform([self.batch_size],dtype=tf.float64)*2*Pi*self.frac_sphere
			ts_in = tf.stack([thetas,phis,psis],axis=-1)
			matrices = TF_RotationBatch(thetas,phis,psis)

			self.xyzs -= self.xyzs[:,0,:][:,tf.newaxis,:]
			self.xyzs_t = tf.einsum('ijk,ikl->ijl', self.xyzs, matrices)
			# Transform the XYZ's
			# The transformation is only WRT the first atom
			# both orig. and transformed system get embedded.

			# Each atom in xyzs gets transformed by ts_in to make t_xyzs
			dxyzs = tf.expand_dims(self.xyzs, axis=2) - tf.expand_dims(self.xyzs, axis=1)
			dxyzs_t = tf.expand_dims(self.xyzs_t, axis=2) - tf.expand_dims(self.xyzs_t, axis=1)
			dist_tensor = tf.norm(dxyzs+1.e-16,axis=3)
			dist_tensor_t = tf.norm(dxyzs_t+1.e-16,axis=3)
			self.Embedded = tf.reshape(tf.reduce_sum(tf_spherical_harmonics(dxyzs, dist_tensor, self.lmax),axis = 2)[:,0,:],[self.batch_size,(self.lmax+1)**2])
			self.Embedded_t = tf.reshape(tf.reduce_sum(tf_spherical_harmonics(dxyzs_t, dist_tensor_t, self.lmax),axis = 2)[:,0,:],[self.batch_size,(self.lmax+1)**2])
			self.EmbeddedShp = tf.shape(self.Embedded)[-1]

			self.Latent, self.Output = self.Capsules(self.Embedded,ts_in)
			self.tLatent, self.tOutput = self.Capsules(self.Embedded_t,tf.zeros_like(ts_in))
			self.dLdR = tf.norm(tf.gradients(self.tLatent, psis))

			self.loss = tf.losses.mean_squared_error(self.Embedded_t, self.Output) + tf.losses.mean_squared_error(self.tLatent, self.Latent)
			tf.add_to_collection('losses', self.loss)
		with tf.name_scope("Adam_optimizer"):
			self.global_step = tf.Variable(0, trainable=False)
			optimizer = tf.train.AdamOptimizer(self.learning_rate)
			tvars = tf.trainable_variables()
			grads_and_vars = optimizer.compute_gradients(self.loss, tvars)
			#clipped = [(tf.clip_by_value(grad, -5, 5), tvar) for grad, tvar in grads_and_vars]
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name="minimize_cost")
		init = tf.global_variables_initializer()
		self.saver = tf.train.Saver(max_to_keep = 10000)
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.sess.run(init)
		self.summary_writer =  tf.summary.FileWriter("./networks/", self.sess.graph)
		return

class Coder:
	"""
	This is an abstract base class for a "coder"
	an object which provides reduced dimensional representations of
	an atom or substructure.
	"""
	def __init__(self,sess_ = None, batch_size_=2000, MaxNAtom_=64, NOutChan_=4):
		"""
		Args:
			NInChan_: Dimension of atom information which will be autoencoded, Ie: 1 for just atomic number.
			NOutChan_: The dimension of the type channel which will result for this atom.
		"""
		self.batch_size = batch_size_
		self.MaxNAtom = MaxNAtom_
		self.out_chan = NOutChan_
		self.learning_rate = 0.002
		self.sess = sess_
	def train(self):
		return
	def decode(self):
		"""
		(VEC=>ZXYZ)
		"""
		return
	def encode(self):
		"""
		This is the key operation provided
		"""
		return
	def loss(self,in_,out_):
		return

class AtomCoder(Coder):
	"""
	An autoencoder which tries to learn a reduced-dimensional embedding
	of an atom by assigning a short vector which 'encodes' the atomic number
	of this atom, by encoding some of it's important properties.

	If you make a plot of the latent vectors with these settings
	Lots of periodic information is encoded in the 4-D latent vector which
	is also quantitatively reversible into the atom's identity.

	There are clear downsides to using this coding (it depends on parameters)
	But hopefully it allows us to treat many atom types.
	"""
	def __init__(self,sess_=None,batch_size_=2000, MaxNAtom_=64, NOutChan_=4):
		self.feature_len = len(AtomData[0][2:])
		# Column normalize the atom data.
		self.AtomData = np.array([AData[2:] for AData in AtomData])
		self.Means = np.zeros(self.feature_len)
		self.Stds = np.zeros(self.feature_len)
		for i in range(self.feature_len):
			self.Means[i] = np.mean(self.AtomData[:,i])
			self.AtomData[:,i] -= self.Means[i]
			self.Stds[i] = np.std(self.AtomData[:,i])
			self.AtomData[:,i] /= self.Stds[i]
		self.code_len = NOutChan_
		Coder.__init__(self, sess_=sess_, batch_size_=batch_size_, MaxNAtom_=MaxNAtom_, NOutChan_=NOutChan_)
		self.Prepare()
		return
	def UnNormalize(self,in_):
		"""
		Args:
			inZs_: an atomic number tensor (NMol X MaxNAtom)
		Returns:
			Supervised Features for this batch.
			Each Atom gets it's vector of atom information
		"""
		return in_*self.stds+self.means
	def SupervisedFeature(self,inZs_):
		"""
		Args:
			inZs_: an atomic number tensor (NMol X MaxNAtom)
		Returns:
			Supervised Features for this batch.
			Each Atom gets it's vector of atom information
		"""
		Z = tf.cast(tf.reshape(inZs_,[self.batch_size*self.MaxNAtom]),tf.int32)
		feat_shape = tf.shape(self.AtomInformation[0])
		mapped_z = tf.gather(self.AtomInformation,Z)
		return mapped_z
	def decode(self,in_):
		"""
		(VEC=>ZXYZ)
		"""
		l1=tf.layers.dense(inputs=in_, units=64, activation=tf.nn.selu,use_bias=True)
		l2=tf.layers.dense(inputs=l1, units=64, activation=tf.nn.selu,use_bias=True)
		l3=tf.layers.dense(inputs=l2, units=self.feature_len, activation=tf.nn.selu,use_bias=True)
		return l3
	def encode(self,in_):
		"""
		This is the key operation provided
		"""
		inr = tf.reshape(in_,[self.batch_size*self.MaxNAtom,self.feature_len])
		l1=tf.layers.dense(inputs=inr, units=64, activation=tf.nn.selu,use_bias=True)
		l2=tf.layers.dense(inputs=l1, units=64, activation=tf.nn.selu,use_bias=True)
		l3=tf.layers.dense(inputs=l2, units=self.code_len, activation=tf.nn.selu,use_bias=True)
		return l3
	def Prepare(self):
		with tf.Graph().as_default():
			with tf.name_scope('AtomCoder'):
				self.AtomInformation = tf.constant(self.AtomData)
				self.means = tf.constant(self.Means)
				self.stds = tf.constant(self.Stds)
				self.Z_pl = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.MaxNAtom])
				self.feat_shape = tf.shape(self.AtomInformation)[1:]
				self.SupervisedOutput = self.SupervisedFeature(self.Z_pl)
				self.RawSupervisedOutput = self.UnNormalize(self.SupervisedOutput)
				self.LatentOutput = self.encode(self.SupervisedOutput)
				self.output = self.decode(self.encode(self.SupervisedOutput))
				self.RawOutput = self.UnNormalize(self.output)
				self.loss = self.loss_op(self.output, self.SupervisedOutput)
				self.train_op = self.training(self.loss)
				if (self.sess==None):
					init = tf.global_variables_initializer()
					self.saver = tf.train.Saver(max_to_keep = 10000)
					self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
					self.sess.run(init)
					self.summary_writer =  tf.summary.FileWriter("./", self.sess.graph)
	def training(self, loss):
		tf.summary.scalar(loss.op.name, loss)
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op
	def loss_op(self,output_,groundtruth_):
		"""
		The goal is to learn Z of this atom, hybridization of this atom
		 Z of nearby atoms, and hybridization of nearby atoms.
		"""
		loss = tf.losses.mean_squared_error(output_,groundtruth_)
		tf.add_to_collection('losses', loss)
		return loss
	def getnextbatch(self):
		"""
		Generate a batch of random integers, and train on them.
		"""
		bzs = np.random.randint(1,MAX_ATOMIC_NUMBER,size=(self.batch_size,self.MaxNAtom))
		return bzs
	def print_training(self, step, loss_):
		if (step%10==0):
			print("step: ", "%7d"%step, "  train loss: ", "%.10f"%(float(loss_)))
		return
	def train_step(self,step):
		data = self.getnextbatch()
		feed_dict = {self.Z_pl:data}
		_,train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
		self.print_training(step, train_loss)
		return
	def train(self, mxsteps=1000):
		test_freq = 40
		for step in range(1, mxsteps+1):
			self.train_step(step)
		return
	def make_emb_factors(self):
		"""
		I really don't think this is how this class should be used.
		instead I think the encoding shoule be trainable with the energy
		but for the time-being, let's just give it a try.
		"""
		tmp = np.zeros((53,self.out_chan))
		for I in range(54):
		    data = np.ones((self.batch_size,self.MaxNAtom))*I
		    feed_dict = {self.Z_pl:data}
		    lout = ac.sess.run([self.LatentOutput], feed_dict=feed_dict)
		    tmp[I] = lout[0][0]
		return tmp

class GeometryCoder(AtomCoder):
	"""
	A geometry is an AtomCoder which additionally compresses
	the hybridization of atoms using HybMatrix(). The idea
	is that this could extend the sight-range of a BP or GauSH.
	"""

class ClusterCoder:
	"""
	This is a compressive encoding of a substructure.
	Separate encodings can be learned for substructure types?
	This is for coarse-graining.
	"""
	def __init__(self):
		return
