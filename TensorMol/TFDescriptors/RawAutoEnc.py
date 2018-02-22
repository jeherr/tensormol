"""
Unlike a symmetry function, an encoder embedding doesn't encode atomic numbers into channels
Ie: it doesn't have pair or triples element channels. Instead it maps every atom onto some basis of vectors unsupervised channels which arise, for example from an autoencoder. These channels are then used in a traditional geometric descriptor (GauSH or SF).
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
from ..ElementData import *
from tensorflow.python.client import timeline
if (HAS_TF):
	import tensorflow as tf

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
