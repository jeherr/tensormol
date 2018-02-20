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

class ChemCoder:
	"""
	This is useful if you want to ensure that your autoencoder is invertible
	to some pieces of chemical information you may decide are critical
	to nearby energy.
	"""


class GeometryCoder:
	"""
	This is an abstract base class for a "Geometry coder"
	an object which provides reduced dimensional representations of
	an atom or substructure. It can be asked to encode possibly
	more than Z, perhaps charges.
	"""
	def __init__(self,batch_size_=2000, NInChan_=1, NOutChan_=4):
		"""
		Args:
			NInChan_: Dimension of atom information which will be autoencoded, Ie: 1 for just atomic number.
			NOutChan_: The dimension of the type channel which will result for this atom.
		"""
		self.batch_size = batch_size_
		self.out_chan = NOutChan_
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

class AtomCoder:
	"""
	An autoencoder which tries to learn a reduced-dimensional invariant embedding
	of an atom by assigning a short vector which 'encodes' the atomic number and hybridization
	of this atom and nearby atoms within RCUT smoothly.
	"""
	def __init__(self,batch_size_=2000, NInChan_=1, NOutChan_=4, RCUT = 2.0):
		GeometryCoder.__init__(self, batch_size_=2000, NInChan_=1, NOutChan_=4, RCUT = 2.0)
		self.AtomInformation = tf.constant(np.array([AData[2:] for AData in AtomData]))
		return
	def SupervisedFeature(self,inXYZ_,inZs_):
		"""
		Args:
			inXYZ_: a set of molecules. NMol X MaxNAtom X 3
			inZs_: an atomic number tensor.
		Returns:
			Supervised Features for this batch.
			Each Atom gets it's vector of atom information, followed
		"""
	def decode(self,in_):
		"""
		(VEC=>ZXYZ)
		"""
		return
	def encode(self,in_):
		"""
		This is the key operation provided
		"""
		return
	def loss(self,in_,out_):
		"""
		The goal is to learn Z of this atom, hybridization of this atom
		 Z of nearby atoms, and hybridization of nearby atoms.
		"""

		return

class ClusterCoder:
	"""
	This is a compressive encoding of a substructure.
	Separate encodings can be learned for substructure types?
	"""
	def __init__(self,RCUT = 2.0):
		return
