"""
An Autoencoded version of a GauSH network.
Which uses an AtomCoder to treat many possible atom types.
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

from ..TFDescriptors.RawAutoEnc import *
from .TFBehlerParinello import *
from tensorflow.python.client import timeline

class BPAutoEncGauSH(BehlerParinelloGauSH):
	def __init__(self, mol_set_name=None, name=None):
		self.atom_coding = PARAMS["atom_coding"]
		BehlerParinelloGauSH.__init__(self, mol_set_name=mol_set_name, name=None)
		self.coder = AtomCoder(batch_size_=self.num_molecules, MaxNAtom_=self.max_num_atoms)
		self.coder.train()
		self.embed_factor = self.coder.make_emb_factors()
		return
