import os
import numpy as np
from ..TFDescriptors.RawSH import *

def sftpluswparam(x):
	return tf.log(1.0+tf.exp(100.*x))/100.0

def safe_inv_norm(x_):
	nrm = tf.clip_by_value(tf.norm(x_,axis=-1,keepdims=True),1e-36,1e36)
	nrm_ok = tf.logical_and(tf.not_equal(nrm,0.),tf.logical_not(tf.is_nan(nrm)))
	safe_nrm = tf.where(nrm_ok,nrm,tf.ones_like(nrm))
	return tf.where(nrm_ok,1.0/safe_nrm,tf.zeros_like(nrm))

def safe_norm(x_):
	nrm = tf.clip_by_value(tf.norm(x_,axis=-1,keepdims=True),1e-36,1e36)
	nrm_ok = tf.logical_and(tf.not_equal(nrm,0.),tf.logical_not(tf.is_nan(nrm)))
	safe_nrm = tf.where(nrm_ok,nrm,tf.zeros_like(nrm))
	return safe_nrm

def polykern(r):
	"""
	Polynomial cutoff 1/r (in BOHR) obeying:
	kern = 1/r at SROuter and LRInner
	d(kern) = d(1/r) (true force) at SROuter,LRInner
	d**2(kern) = d**2(1/r) at SROuter and LRInner.
	d(kern) = 0 (no force) at/beyond SRInner and LROuter

	The hard cutoff is LROuter
	"""
	# This one is pretty slutty, 11A cutoff.
	if 0:
		SRInner = 4.5*1.889725989
		SROuter = 6.5*1.889725989
		LRInner = 10.*1.889725989
		LROuter = 11.*1.889725989
		a = -73.6568
		b = 37.1829
		c = -7.86634
		d = 0.9161
		e = -0.0634132
		f = 0.00260868
		g = -0.0000590516
		h = 5.67472e-7
	if 0:
		SRInner = 4.5*1.889725989
		SROuter = 6.5*1.889725989
		LRInner = 13.*1.889725989
		LROuter = 15.*1.889725989
		a = -18.325
		b = 7.89288
		c = -1.35458
		d = 0.126578
		e = -0.00695692
		f = 0.000225092
		g = -3.97476e-6
		h = 2.95926e-8
	if 0:
		SRInner = 6.0
		SROuter = 8.0
		LRInner = 13.
		LROuter = 15.
		a = -57.6862
		b = 40.2721
		c = -11.7317
		d = 1.8806
		e = -0.179183
		f = 0.0101501
		g = -0.000316629
		h = 4.19773e-6
	SRInner = 6.0
	SROuter = 9.0
	LRInner = 16.
	LROuter = 19.
	a = -17.8953
	b = 11.0312
	c = -2.72193
	d = 0.36794
	e = -0.0294324
	f = 0.00139391
	g = -0.0000362146
	h = 3.98488e-7
	r2=r*r
	r3=r2*r
	r4=r3*r
	r5=r4*r
	r6=r5*r
	r7=r6*r
	kern = tf.where(tf.greater(r,LROuter),
				1.0/LROuter*tf.ones_like(r),
				tf.where(tf.less(r,SRInner),
				1.0/SRInner*tf.ones_like(r),
				(a+b*r+c*r2+d*r3+e*r4+f*r5+g*r6+h*r7)/r))
	return kern

class SparseCodedChargedGauSHNetwork:
	"""
	This is the basic TensorMol0.2 model chemistry.
	"""
	def __init__(self,aset=None,load=False,load_averages=False,mode='train'):
		self.prec = tf.float64
		self.batch_size = 24
		self.MaxNAtom = 32
		self.MaxNeigh_NN = self.MaxNAtom
		self.MaxNeigh_J = self.MaxNAtom
		self.learning_rate = 0.00005
		self.ncan = 6
		self.DoHess=False
		self.mode = mode
		self.Lsq = False
		if (mode == 'eval'):
			self.ncan = 6
		self.RCut_Coulomb = 19.0
		self.RCut_NN = 7.0
		self.AtomCodes = ELEMENTCODES
		#self.AtomCodes = np.random.random(size=(PARAMS["MAX_ATOMIC_NUMBER"],6))
		self.AtomTypes = [1,6,7,8]
		self.l_max = 4
		self.GaussParams = np.array([[0.41032325, 0.27364972],
			[0.97418311, 0.25542902],
			[1.53804298, 0.23720832],
			[2.10190284, 0.21898761],
			[2.66576271, 0.20076691],
			[3.22962257, 0.18254621],
			[3.79348244, 0.16432551],
			[4.3573423 , 0.1461048 ],
			[4.92120217, 0.1278841 ],
			[5.48506203, 0.1096634 ]])
		self.nrad = len(self.GaussParams)
		self.nang = (self.l_max+1)**2
		if (self.Lsq):
			self.nang = self.l_max
		self.ncodes = self.AtomCodes.shape[-1]
		self.ngaush = self.nrad*self.nang
		self.nembdim = self.ngaush*self.ncodes
		self.mset = aset
		self.AverageElementEnergy = np.zeros((PARAMS["MAX_ATOMIC_NUMBER"]))
		self.AverageElementCharge = np.zeros((PARAMS["MAX_ATOMIC_NUMBER"]))
		if (aset != None):
			self.MaxNAtom = aset.MaxNAtom()+1
			self.AtomTypes = aset.AtomTypes()
			AvE,AvQ = aset.RemoveElementAverages()
			for ele in AvE.keys():
				self.AverageElementEnergy[ele] = AvE[ele]
				self.AverageElementCharge[ele] = AvQ[ele]
			# Get a number of neighbors.
			# Looking at the molecules with the most atoms.
			xyzs = np.zeros((50,self.MaxNAtom,3))
			zs = np.zeros((50,self.MaxNAtom,1),dtype=np.int32)
			natoms = [m.NAtoms() for m in aset.mols]
			mol_order = np.argsort(natoms)[::-1]
			for i in range(min(50,len(aset.mols))):
				mi = mol_order[i]
				m = aset.mols[mi]
				xyzs[i,:m.NAtoms()] = m.coords
				zs[i,:m.NAtoms(),0] = m.atoms
			NL_NN, NL_J = self.NLTensors(xyzs,zs)
			self.MaxNeigh_NN = NL_NN.shape[-1]+2
			self.MaxNeigh_J = NL_J.shape[-1]+2

			# The maximum number of non-negative indices in dimension 2
			#print("self.MaxNeigh_NN = ", self.MaxNeigh_NN)
			#print("self.MaxNeigh_J = ", self.MaxNeigh_J)
		self.sess = None
		self.Prepare()
		if (load):
			self.Load(load_averages)
		return

	def Load(self,load_averages=False):
		try:
			chkpt = tf.train.latest_checkpoint('./networks/')
			metafilename = ".".join([chkpt, "meta"])
			if (self.sess == None):
				self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#print("Loading Graph: ",metafilename)
			#saver = tf.train.import_meta_graph(metafilename)
			self.saver.restore(self.sess,chkpt)
			print("Loaded Check: ",chkpt)
			# Recover key variables.
			try:
				self.AtomCodes = self.sess.run([v for v in tf.global_variables() if v.name == "atom_codes:0"][0])
				self.GaussParams = self.sess.run([v for v in tf.global_variables() if v.name == "gauss_params:0"][0])
			except:
				self.AtomCodes = self.sess.run([v for v in tf.global_variables() if v.name == "Variable:0"][0])
				self.GaussParams = self.sess.run([v for v in tf.global_variables() if v.name == "Variable_1:0"][0])
			# if these are None, Restore them.
			if (not type(self.AverageElementEnergy) is np.ndarray or load_averages):
				try:
					self.AverageElementEnergy = self.sess.run([v for v in tf.global_variables() if v.name == "av_energies:0"][0])
					self.AverageElementCharge = self.sess.run([v for v in tf.global_variables() if v.name == "av_charges:0"][0])
				except:
					self.AverageElementEnergy = self.sess.run([v for v in tf.global_variables() if v.name == "Variable_2:0"][0])
					self.AverageElementCharge = self.sess.run([v for v in tf.global_variables() if v.name == "Variable_3:0"][0])
				print("self.AvE", self.AverageElementEnergy)
				print("self.AvQ", self.AverageElementCharge)
		except Exception as Ex:
			print("Load failed.",Ex)
			raise Ex
		return

	def GetEnergyForceRoutine(self,m,Debug=False):
		MustPrepare = False
		if (self.batch_size>10):
			self.batch_size=1
			MustPrepare=True
		if (m.NAtoms() > self.MaxNAtom):
			self.MaxNAtom = m.NAtoms()
			MustPrepare=True
			self.batch_size=1
		xyzs_t = np.zeros((self.batch_size,self.MaxNAtom,3))
		Zs_t = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
		xyzs_t[0,:m.NAtoms(),:] = m.coords
		Zs_t[0,:m.NAtoms(),0] = m.atoms
		if ((self.MaxNeigh_J > self.MaxNeigh_J_prep) or (self.MaxNeigh_NN > self.MaxNeigh_NN_prep)):
			self.batch_size=1
			MustPrepare=True
		if (MustPrepare):
			self.Prepare()
			self.Load()
		self.NEFCalls = 0
		self.CAxes = np.zeros((self.ncan,self.batch_size*self.MaxNAtom,3,3))
		self.CWeights = np.zeros((self.ncan,self.batch_size,self.MaxNAtom))
		self.CWeights, self.CAxes = self.sess.run([self.CanonicalAxes(self.dxyzs,self.sparse_mask)],feed_dict=self.MakeFeed(Mol(m.atoms,m.coords)))[0]
		def EF(xyz_,DoForce=True,Debug = False):
			self.NEFCalls += 1
			feed_dict = self.MakeFeed(Mol(m.atoms,xyz_))
			#if (self.NEFCalls%100==0):
			#	self.CWeights, self.CAxes = self.sess.run([self.CanonicalAxes(self.dxyzs,self.sparse_mask)],feed_dict=self.MakeFeed(Mol(m.atoms,xyz_)))[0]
			if (self.DoRotGrad):
				print("RotGrad:",self.sess.run([self.RotGrad], feed_dict=feed_dict))
			if (Debug):
				print(nls_nn,nls_j)
			if (DoForce):
				ens,fs = self.sess.run([self.MolEnergies,self.MolGrads], feed_dict=feed_dict)
				ifrc = RemoveInvariantForce(xyz_,fs[0][:m.NAtoms()],m.MassVector())
				return ens[0],ifrc*(-JOULEPERHARTREE)
			else:
				ens = self.sess.run(self.MolEnergies, feed_dict=feed_dict)[0]
				return ens[0]
		return EF

	def GetBatchedEnergyForceRoutine(self,mset,Debug=False):
		self.batch_size=len(mset.mols)
		m = mset.mols[0]
		MustPrepare=True
		if (mset.MaxNAtom() > self.MaxNAtom):
			self.MaxNAtom = mset.MaxNAtom()
		xyzs_t = np.zeros((self.batch_size,self.MaxNAtom,3))
		Zs_t = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
		for i in range(self.batch_size):
			xyzs_t[i,:,:] = mset.mols[i].coords
			Zs_t[i,:,0] = mset.mols[i].atoms
		nlt_nn, nlt_j = self.NLTensors(xyzs_t,Zs_t)
		if ((self.MaxNeigh_J > self.MaxNeigh_J_prep) or (self.MaxNeigh_NN > self.MaxNeigh_NN_prep)):
			self.batch_size=len(mset.mols)
			MustPrepare=True
		if (MustPrepare):
			self.Prepare()
			self.Load()
		self.CAxes = np.zeros((self.ncan,self.batch_size*self.MaxNAtom,3,3))
		self.CWeights = np.zeros((self.ncan,self.batch_size,self.MaxNAtom))

		nls_nn = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_NN),dtype=np.int32)
		nls_j = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_J),dtype=np.int32)
		nls_nn[:nlt_nn.shape[0],:nlt_nn.shape[1],:nlt_nn.shape[2]] = nlt_nn
		nls_j[:nlt_j.shape[0],:nlt_j.shape[1],:nlt_j.shape[2]] = nlt_j

		FD = {self.xyzs_pl:xyzs_t, self.zs_pl:Zs_t, self.nl_nn_pl:nls_nn,self.nl_j_pl:nls_j}
		self.CWeights, self.CAxes = self.sess.run([self.CanonicalAxes(self.dxyzs,self.sparse_mask)],feed_dict=FD)[0]
		for k in range(self.CWeights.shape[0]):
			self.CWeights[k] = self.CWeights[0]
			self.CAxes[k] = self.CAxes[0]

		def EF(xyz_,DoForce=True,Debug = False):
			xyzs = np.zeros((self.batch_size,self.MaxNAtom,3))
			Zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
			nls_nn = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_NN),dtype=np.int32)
			nls_j = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_J),dtype=np.int32)
			xyzs[:,:m.NAtoms(),:] = xyz_
			for i in range(self.batch_size):
				Zs[i,:,0] = m.atoms
			nlt_nn, nlt_j = self.NLTensors(xyzs,Zs)
			if ((self.MaxNeigh_J > self.MaxNeigh_J_prep) or (self.MaxNeigh_NN > self.MaxNeigh_NN_prep)):
				nls_nn = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_NN),dtype=np.int32)
				nls_j = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_J),dtype=np.int32)
				self.Prepare()
				self.Load()
			nls_nn[:nlt_nn.shape[0],:nlt_nn.shape[1],:nlt_nn.shape[2]] = nlt_nn
			nls_j[:nlt_j.shape[0],:nlt_j.shape[1],:nlt_j.shape[2]] = nlt_j
			feed_dict = {self.xyzs_pl:xyzs,
					self.zs_pl:Zs,
					self.nl_nn_pl:nls_nn,
					self.nl_j_pl:nls_j,
					self.cax_pl:self.CAxes,
					self.cw_pl:self.CWeights}
			if (Debug):
				print(nls_nn,nls_j)
			if (DoForce):
				ens,fs = self.sess.run([self.MolEnergies,self.MolGrads], feed_dict=feed_dict)
				return ens,fs[:,:m.NAtoms(),:]*(-JOULEPERHARTREE)
			else:
				ens = self.sess.run(self.MolEnergies, feed_dict=feed_dict)[0]
				return ens
		return EF

	def GetEnergyForceHessRoutine(self,m):
		if (m.NAtoms() > self.MaxNAtom):
			self.MaxNAtom = m.NAtoms()
			self.batch_size=1
		xyzs = np.zeros((self.batch_size,self.MaxNAtom,3))
		Zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
		xyzs[0,:m.NAtoms(),:] = m.coords
		Zs[0,:m.NAtoms(),0] = m.atoms
		nlt_nn, nlt_j = self.NLTensors(xyzs_t,Zs_t)
		self.DoHess=True
		self.Prepare()
		self.Load()
		def EFH(xyz_):
			xyzs = np.zeros((self.batch_size,self.MaxNAtom,3))
			Zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
			nls_nn = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_NN),dtype=np.int32)
			nls_j = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_J),dtype=np.int32)
			xyzs[0,:m.NAtoms(),:] = xyz_
			Zs[0,:m.NAtoms(),0] = m.atoms
			nlt,MaxNeigh = self.NLTensors(xyzs,Zs)
			if (MaxNeigh > self.MaxNeigh):
				print("Too Many Neighbors.")
				raise Exception('NeighborOverflow')
			nls[:nlt.shape[0],:nlt.shape[1],:nlt.shape[2]] = nlt
			feed_dict = {self.xyzs_pl:xyzs, self.zs_pl:Zs,self.nl_nn_pl:nls_nn,self.nl_j_pl:nls_j}
			ens,fs,hs = self.sess.run([self.MolEnergies,self.MolGrads,self.MolHess], feed_dict=feed_dict)
			return ens[0], fs[0][:m.NAtoms()]*(-JOULEPERHARTREE), hs[0][:m.NAtoms()][:m.NAtoms()]*JOULEPERHARTREE*JOULEPERHARTREE
		return EFH

	@TMTiming("MakeFeed")
	def MakeFeed(self,m):
		"""
 		Randomly accumulate a batch.

		Args:
			aset: A molecule set.

		Returns:
 			feed dictionary and the molecule tested for debug purposes.
		"""
		xyzs = np.zeros((self.batch_size,self.MaxNAtom,3),dtype=np.float)
		true_force = np.zeros((self.batch_size,self.MaxNAtom,3),dtype=np.float)
		zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
		true_ae = np.zeros((self.batch_size,1),dtype=np.float)
		qs = np.zeros((self.batch_size,self.MaxNAtom),dtype=np.float64) # Charges.
		ds = np.zeros((self.batch_size,3),dtype=np.float64) # Dipoles.
		xyzs[0,:m.NAtoms()] = m.coords
		zs[0,:m.NAtoms(),0] = m.atoms
		nlt_nn, nlt_j = self.NLTensors(xyzs,zs)
		nls_nn = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_NN),dtype=np.int32)
		nls_j = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_J),dtype=np.int32)
		if ((self.MaxNeigh_J > self.MaxNeigh_J_prep) or (self.MaxNeigh_NN > self.MaxNeigh_NN_prep)):
			self.Prepare()
			self.Load()
		nls_nn[:nlt_nn.shape[0],:nlt_nn.shape[1],:nlt_nn.shape[2]] = nlt_nn
		nls_j[:nlt_j.shape[0],:nlt_j.shape[1],:nlt_j.shape[2]] = nlt_j
		return {self.xyzs_pl:xyzs,
				self.zs_pl:zs,
				self.nl_nn_pl:nls_nn,
				self.nl_j_pl:nls_j,
				self.cax_pl:self.CAxes,
				self.cw_pl:self.CWeights,
				self.groundTruthE_pl:true_ae,
				self.groundTruthG_pl:true_force,
				self.groundTruthQ_pl:qs,
				self.groundTruthD_pl:ds}

	@TMTiming("NextBatch")
	def NextBatch(self,aset):
		"""
 		Randomly accumulate a batch.

		Args:
			aset: A molecule set.

		Returns:
 			feed dictionary and the molecule tested for debug purposes.
		"""
		xyzs = np.zeros((self.batch_size,self.MaxNAtom,3),dtype=np.float)
		true_force = np.zeros((self.batch_size,self.MaxNAtom,3),dtype=np.float)
		zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
		true_ae = np.zeros((self.batch_size,1),dtype=np.float)
		qs = np.zeros((self.batch_size,self.MaxNAtom),dtype=np.float64) # Charges.
		ds = np.zeros((self.batch_size,3),dtype=np.float64) # Dipoles.
		mols = []
		i=0
		while i < self.batch_size:
			mi = np.random.randint(len(aset.mols))
			m = aset.mols[mi]
			nancoords = np.any(np.isnan(m.coords))
			nanatoms = np.any(np.isnan(m.atoms))
			nanenergy = np.isnan(m.properties["energy"])
			nangradients = np.any(np.isnan(m.properties["gradients"]))
			nancharges = np.any(np.isnan(m.properties["charges"]))
			if (nancoords or nanatoms or nanenergy or nangradients or nancharges):
				continue
			xyzs[i,:m.NAtoms()] = m.coords
			zs[i,:m.NAtoms(),0] = m.atoms
			true_ae[i]=m.properties["energy"]
			true_force[i,:m.NAtoms()]=m.properties["gradients"]
			qs[i,:m.NAtoms()]=m.properties["charges"]
			try:
				ds[i]=m.properties["dipole"]
			except:
				pass
			i += 1
			mols.append(m)
		nlt_nn, nlt_j = self.NLTensors(xyzs,zs)
		# all atoms need at least two neighbors.
		if (np.any(np.logical_and(np.less(nlt_nn[:,:,1],0),np.greater_equal(nlt_nn[:,:,0],0)))):
			print("... Not enough neighbors in your data ...")
			return self.NextBatch(aset)
		nls_nn = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_NN),dtype=np.int32)
		nls_j = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_J),dtype=np.int32)
		if ((self.MaxNeigh_J > self.MaxNeigh_J_prep) or (self.MaxNeigh_NN > self.MaxNeigh_NN_prep)):
			self.Prepare()
			self.Load()
		nls_nn[:nlt_nn.shape[0],:nlt_nn.shape[1],:nlt_nn.shape[2]] = nlt_nn
		nls_j[:nlt_j.shape[0],:nlt_j.shape[1],:nlt_j.shape[2]] = nlt_j
		return {self.xyzs_pl:xyzs, self.zs_pl:zs,self.nl_nn_pl:nls_nn,self.nl_j_pl:nls_j,self.groundTruthE_pl:true_ae, self.groundTruthG_pl:true_force, self.groundTruthQ_pl:qs, self.groundTruthD_pl:ds}, mols

	def Embed(self, dxyzs, jcodes, pair_mask, gauss_params, l_max):
		"""
		Returns the GauSH embedding of every atom.
		as a mol X maxNAtom X ang X rad X code tensor.

		Args:
			dxyzs: (nmol X maxnatom X maxneigh x 3) difference vector.
			jcodes: (nmol X maxnatom x maxneigh X 4) atomic code tensor for atom j
			pair_mask: (nmol X maxnatom X maxneigh x 1) multiplicative mask.
			gauss_params: (nrad X 2) tensor of gaussian paramters.  (ang.)
			l_max: max angular momentum of embedding.
		"""
		dist_tensor = tf.clip_by_value(tf.norm(dxyzs+1.e-36,axis=-1),1e-36,1e36)
		# NMOL X MAXNATOM X MAXNATOM X NSH
		SH = tf_spherical_harmonics(dxyzs, dist_tensor, l_max)*pair_mask # mol X maxNatom X maxNeigh X nang.
		RAD = tf_gauss(dist_tensor, gauss_params)*pair_mask # mol X maxNatom X maxNeigh X nrad.
		# Perform each of the contractions.
		SHRAD = tf.einsum('mijk,mijl->mijkl',SH,RAD) # mol X maxnatom X maxneigh X nang X nrad
		#CODES = tf.reshape(tf.gather(elecode, Zs, axis=0),(self.batch_size,self.MaxNAtom,self.ncodes)) # mol X maxNatom X 4
		SHRADCODE = tf.einsum('mijkl,mijn->mikln',SHRAD,jcodes)
		return SHRADCODE

	@TMTiming("NLTensors")
	def NLTensors(self, xyzs_, zs_, ntodo_=None):
		"""
		Generate Neighborlist arrays for sparse version.

		Args:
			xyzs_ : a coordinate tensor nmol X Maxnatom X 3
			Zs_ : a AN tensor nmol X Maxnatom X 1
			ntodo_: For periodic (unused.)
		Returns:
			nlarray a nmol X Maxnatom X MaxNeighbors int32 array.
				which is -1 = blank, 0 = atom zero is a neighbor within Rcut

			and the number of maximum neighbors found in its argument.
		"""
		nlt_nn = Make_NLTensor(xyzs_,zs_.astype(np.int32),self.RCut_NN, self.MaxNAtom, True, True)
		nlt_j = Make_NLTensor(xyzs_,zs_.astype(np.int32),self.RCut_Coulomb, self.MaxNAtom, False, False)
		self.MaxNeigh_NN = max(nlt_nn.shape[-1]+2,self.MaxNeigh_NN)
		self.MaxNeigh_J = max(nlt_j.shape[-1]+2,self.MaxNeigh_J)
		return nlt_nn, nlt_j

	def CoulombAtomEnergies(self,dxyzs_,q1q2s_):
		"""
		Atom Coulomb energy with polynomial cutoff.
		Zero of energy is set so that if all atom pairs are outside the cutoff the energy is zero.
		Note: No factor of two because the j-neighborlist is sorted.

		Args:
			dxyzs_: (nmol X MaxNAtom X MaxNeigh X 1) distances tensor (angstrom)
			# NOTE: This has no self-energy because of the way dxyzs_ is neighbored.
			q1q2s_: charges (nmol X MaxNAtom X MaxNeigh X 1) charges tensor. (atomic)
		Returns:
			 (nmol X 1) tensor of energies (atomic)
		"""
		EMR = tf.reduce_sum(polykern(dxyzs_*1.889725989)*q1q2s_,axis=(2))[:,:,tf.newaxis]
		KLR = tf.ones_like(dxyzs_)*(1.0/self.RCut_Coulomb)
		ELR = tf.reduce_sum(KLR*q1q2s_,axis=(2))[:,:,tf.newaxis]
		return EMR-ELR

	def ChargeToDipole(self,xyzs_,zs_,qs_):
		"""
		Calculate the dipole moment relative to center of atom.
		"""
		n_atoms = tf.clip_by_value(tf.cast(tf.reduce_sum(zs_,axis=(1,2)),self.prec),1e-36,1e36)
		COA = tf.reduce_sum(xyzs_,axis=1)/n_atoms[:,tf.newaxis]
		return tf.reduce_sum((xyzs_ - COA[:,tf.newaxis,:])*qs_[:,:,tf.newaxis],axis=1)

	def CanonicalAxes(self, dxyzs, sparse_mask):
		"""
		This version returns two sets of axes for nearest and next-nearest neighbor.
		If the energy from both these representations is averaged the result
		will be permutationally invariant (WRT nearest-next-nearest motion)
		and rotationally invariant.

		Args:
		        dxyz: a nMol X maxNatom X maxNatom X 3 tensor of atoms. (differenced from center of embedding
		        zs: a nMol X maxNatom X maxNatom X 1 tensor of atomic number pairs.
		        ie: ... X i X i = (0.,0.,0.))

		        also an ncan X nmol X maxNAtom X 1 tensor
		"""
		# Append orthogonal axes to dxyzs
		argshape = tf.shape(dxyzs)
		realdata = tf.reshape(dxyzs,(argshape[0]*argshape[1],argshape[2],3))
		msk = tf.reshape(sparse_mask,(argshape[0]*argshape[1],argshape[2],1))
		axis_cutoff = self.RCut_NN*self.RCut_NN
		orders=[]
		if (self.ncan == 1):
			orders = [[0,1]]
		elif (self.ncan == 3):
			orders = [[0,1],[1,2],[0,2]]
		elif (self.ncan == 6):
			orders = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3]]
		elif (self.ncan == 10):
			orders = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3],[0,4],[1,4],[2,4],[3,4]]
		elif (self.ncan == 15):
			orders = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3],[0,4],[1,4],[2,4],[3,4],[0,5],[1,5],[2,5],[3,5],[4,5]]
		elif (self.ncan == 21):
			orders = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3],[0,4],[1,4],[2,4],[3,4],[0,5],[1,5],[2,5],[3,5],[4,5],[0,6],[1,6],[2,6],[3,6],[4,6],[5,6]]

		if (self.ncan == 2):
			orders = [[0,1]]
		elif (self.ncan == 6):
			orders = [[0,1],[1,2],[0,2]]
		elif (self.ncan == 12):
			orders = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3]]
		elif (self.ncan == 20):
			orders = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3],[0,4],[1,4],[2,4],[3,4]]
		elif (self.ncan == 30):
			orders = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3],[0,4],[1,4],[2,4],[3,4],[0,5],[1,5],[2,5],[3,5],[4,5]]
		elif (self.ncan == 42):
			orders = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3],[0,4],[1,4],[2,4],[3,4],[0,5],[1,5],[2,5],[3,5],[4,5],[0,6],[1,6],[2,6],[3,6],[4,6],[5,6]]

		weightstore = []
		cuttore = []
		axtore = []

		for perm in orders:
			v1 = tf.reshape(dxyzs[:,:,perm[0],:],(argshape[0]*argshape[1],3))+tf.constant(np.array([1e-23,0.,0.]),dtype=tf.float64)
			v2 = tf.reshape(dxyzs[:,:,perm[1],:],(argshape[0]*argshape[1],3))+tf.constant(np.array([0.,1e-23,0.]),dtype=tf.float64)
			w1 = tf.reshape(tf.reduce_sum(dxyzs[:,:,perm[0],:]*dxyzs[:,:,perm[0],:],axis=-1),(argshape[0]*argshape[1],1))
			w2 = tf.reshape(tf.reduce_sum(dxyzs[:,:,perm[1],:]*dxyzs[:,:,perm[1],:],axis=-1),(argshape[0]*argshape[1],1))

			v1n = safe_inv_norm(v1)*v1
			v2n = safe_inv_norm(v2)*v2
			w3p = tf.abs(tf.reduce_sum(v1n*v2n,axis=-1)[...,tf.newaxis])

			v3 = tf.cross(v1n,v2n)
			#posz = tf.tile(tf.greater(tf.reduce_sum(v3refl*tf.constant([[1.,1.,1.]],dtype=self.prec),keepdims=True,axis=-1),0.),[1,3])
			#v3 = tf.where(posz,v3refl,-1.*v3refl)
			v3 *= safe_inv_norm(v3)

			# Compute the average of v1, v2, and their projections onto the plane.
			v_av = (v1n+v2n)/2.0
			v_av *= safe_inv_norm(v_av)

			# Rotate pi/4 cw and ccw to obtain v1,v2
			first = TF_AxisAngleRotation(v3,v_av,tf.constant(Pi/4.,dtype=self.prec))
			second = TF_AxisAngleRotation(v3,v_av,tf.constant(-Pi/4.,dtype=self.prec))

			vs = tf.concat([first[:,tf.newaxis,:],second[:,tf.newaxis,:],v3[:,tf.newaxis,:]],axis=1)

			cw1 = tf.where(tf.less(w1,axis_cutoff),tf.cos(w1/axis_cutoff*Pi/2.0),tf.zeros_like(w1))
			cw2 = tf.where(tf.less(w2,axis_cutoff),tf.cos(w2/axis_cutoff*Pi/2.0),tf.zeros_like(w2))
			ca = tf.where(tf.greater(w3p,0.92),tf.zeros_like(w3p),tf.cos(w3p/0.92*Pi/2.))

			Cutoffs = cw1*cw2*ca*msk[:,perm[0],:]*msk[:,perm[1],:]
			weightstore.append(w1+w2)
			cuttore.append(Cutoffs)
			axtore.append(vs)

			vs2 = tf.concat([second[:,tf.newaxis,:],first[:,tf.newaxis,:],-1*v3[:,tf.newaxis,:]],axis=1)
			weightstore.append(w1+w2)
			cuttore.append(Cutoffs)
			axtore.append(vs2)

		Cuts = tf.stack(cuttore,axis=0)
		safeweights = tf.clip_by_value(tf.stack(weightstore,axis=0),1e-19,36.0)
		#pw = tf.nn.softmax(-1*tf.stack(weightstore,axis=0),axis=0)*Cuts
		pw = Cuts*tf.exp(-safeweights)#/(tf.stack(weightstore,axis=0) + 1e-9)
		tdn = tf.where(tf.greater_equal(Cuts,0.), pw, tf.zeros_like(pw))
		dn = tf.clip_by_value(tf.reduce_sum(tdn,axis=0,keepdims=True),1e-19,1e19)
		tw = tf.where(tf.greater_equal(Cuts,0.), tdn/(dn), tf.zeros_like(tdn))
		weights = tf.reshape(tw,(self.ncan,argshape[0],argshape[1]))
		return weights, tf.stack(axtore,axis=0)

	def ChargeEmbeddedModel(self, dxyzs, Zs, zjs, gather_inds, pair_mask, gauss_params, atom_codes, l_max):
		"""
		This version creates a network to integrate weight information.
		and then works like any ordinary network.
		NOTE: This network is universal in the sense that it works on ANY atom!

		Args:
			dxyzs: (nmol X maxnatom X maxneigh x 3) difference vector.
			Zs: mol X maxNatom X 1 atomic number tensor.
			zjs: nmol X maxnatom x maxneigh X 1 neighbor atomic numbers.
			gather_inds: neighbor indices.
			pair_mask: (nmol X maxnatom X maxneigh x 1) multiplicative mask.
			gauss_params: (nrad X 2) tensor of gaussian paramters.  (ang.)
			l_max: max angular momentum of embedding.
		Returns:
			mol X maxNatom X 1 tensor of atom energies, charges.
			these include constant shifts.
		"""
		ncase = self.batch_size*self.MaxNAtom
		nchan = self.AtomCodes.shape[1]

		Zrs = tf.cast(tf.reshape(Zs,(ncase,-1)),self.prec)
		Atom12Real = tf.not_equal(pair_mask,0.)
		Atom12Real4 = tf.tile(Atom12Real,[1,1,1,nchan])
		Atom12Real5 = tf.tile(Atom12Real,[1,1,1,nchan+1])

		with tf.variable_scope("AtomVariance", reuse=tf.AUTO_REUSE):
			stdinit = tf.constant(np.ones(PARAMS["MAX_ATOMIC_NUMBER"]),dtype=self.prec)
			self.AtomEStd = tf.get_variable(name="AtomEStd",dtype=self.prec,initializer=stdinit)
			AtomEStds = tf.reshape(tf.gather(self.AtomEStd, Zs, axis=0),(self.batch_size,self.MaxNAtom,1))

		jcodes0 = tf.reshape(tf.gather(atom_codes,zjs),(self.batch_size,self.MaxNAtom,self.MaxNeigh_NN,nchan))
		jcodes = tf.where(Atom12Real4 , jcodes0 , tf.zeros_like(jcodes0))# mol X maxNatom X maxnieh X 4

		# construct embedding.
		dist_tensor = tf.clip_by_value(tf.norm(dxyzs+1.e-36,axis=-1),1e-36,1e36)
		# NMOL X MAXNATOM X MAXNATOM X NSH
		SH = tf_spherical_harmonics(dxyzs, dist_tensor, l_max, invariant = self.Lsq)*pair_mask # mol X maxNatom X maxNeigh X nang.
		RAD = tf_gauss(dist_tensor, gauss_params)*pair_mask # mol X maxNatom X maxNeigh X nrad.
		# Perform each of the contractions.
		SHRAD = tf.einsum('mijk,mijl->mijkl',SH,RAD) # mol X maxnatom X maxneigh X nang X nrad
		#CODES = tf.reshape(tf.gather(elecode, Zs, axis=0),(self.batch_size,self.MaxNAtom,self.ncodes)) # mol X maxNatom X 4
		emb = tf.einsum('mijkl,mijn->mikln',SHRAD,jcodes)

		# Get codes and averages of atom i
		AvEs = tf.reshape(tf.gather(self.AvE_tf, Zs, axis=0),(self.batch_size,self.MaxNAtom,1)) # (mol * maxNatom) X 1
		AvQs = tf.reshape(tf.gather(self.AvQ_tf, Zs, axis=0),(self.batch_size,self.MaxNAtom)) # (mol * maxNatom) X 1
		CODES = tf.reshape(tf.gather(self.atom_codes, Zs, axis=0),(ncase,nchan)) # (mol * maxNatom) X 4

		# the idea for the new network is to keep the code dimension
		# for at least the first layers.

		# Combine the codes of the main atom and the sensed atom
		# Using a hinton-esque tensor decomposition.
		with tf.variable_scope("chargenet", reuse=tf.AUTO_REUSE):
			CODEKERN1 = tf.get_variable(name="CodeKernel", shape=(nchan,nchan),dtype=self.prec)
			CODEKERN2 = tf.get_variable(name="CodeKernel2", shape=(nchan,nchan),dtype=self.prec)
			# combine the weight kernel with the codes.
			mix1 = tf.matmul(CODES,CODEKERN1) # ncase X ncode
			embrs = tf.reshape(emb,(ncase,-1,nchan))
			# Ensure any zero cases don't contribute.
			msk = tf.where(tf.equal(Zrs,0.0),tf.zeros_like(Zrs),tf.ones_like(Zrs))
			embrs *= msk[:,:,tf.newaxis]
			weighted = tf.einsum('ikj,ij->ikj',embrs,mix1)
			weighted2 = tf.einsum('ikj,jl->ikl',weighted,CODEKERN2)
			# Now pass it through as usual.
			l0 = tf.reshape(weighted2,(ncase,-1))
			l0p = tf.concat([l0,CODES],axis=-1)

			l1q = tf.layers.dense(inputs=l0p,units=512,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense1q")
			l1pq = tf.concat([l1q,CODES],axis=-1)
			l2q = tf.layers.dense(inputs=l1pq,units=512,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense2q")
			l2pq = tf.concat([l2q,CODES],axis=-1)
			l3q = tf.layers.dense(l2pq,units=1,activation=None,use_bias=False,name="Dense3q")*msk
			charges = tf.reshape(l3q,(self.batch_size,self.MaxNAtom))
			# Set the total charges to neutral by evenly distributing any excess charge.
			excess_charges = tf.reduce_sum(charges,axis=[1])
			n_atoms = tf.reduce_sum(tf.where(tf.equal(Zs,0),Zs,tf.ones_like(Zs)),axis=[1,2])
			fix = -1.0*excess_charges/tf.cast(n_atoms,self.prec)
			AtomCharges = charges + fix[:,tf.newaxis] + AvQs

		# Now concatenate the charges onto the embedding for the energy network.
		with tf.variable_scope("energynet", reuse=tf.AUTO_REUSE):
			qcodes = tf.reshape(tf.gather_nd(AtomCharges, gather_inds),(self.batch_size,self.MaxNAtom,self.MaxNeigh_NN,1))
			jcodes0_wq = tf.concat([jcodes,qcodes],axis=-1)
			jcodes_wq = tf.where(Atom12Real5 , jcodes0_wq , tf.zeros_like(jcodes0_wq))# mol X maxNatom X maxnieh X 4
			emb_wq = tf.einsum('mijkl,mijn->mikln',SHRAD,jcodes_wq)
			CODES_wq = tf.concat([CODES,tf.reshape(AtomCharges,(ncase,1))],axis=-1)
			CODEKERN1_wq = tf.get_variable(name="CodeKernel_wq", shape=(nchan+1,nchan+1),dtype=self.prec)
			CODEKERN2_wq = tf.get_variable(name="CodeKernel2_wq", shape=(nchan+1,nchan+1),dtype=self.prec)
			# combine the weight kernel with the codes.
			mix1_wq = tf.matmul(CODES_wq,CODEKERN1_wq) # ncase X ncode
			embrs_wq = tf.reshape(emb_wq,(ncase,-1,nchan+1))
			# Ensure any zero cases don't contribute.
			msk_wq = tf.where(tf.equal(Zrs,0.0),tf.zeros_like(Zrs),tf.ones_like(Zrs))
			embrs_wq *= msk_wq[:,:,tf.newaxis]
			weighted_wq = tf.einsum('ikj,ij->ikj',embrs_wq,mix1_wq)
			weighted2_wq = tf.einsum('ikj,jl->ikl',weighted_wq,CODEKERN2_wq)
			# Now pass it through as usual.
			l0_wq = tf.reshape(weighted2_wq,(ncase,-1))
			l0p_wq = tf.concat([l0_wq,CODES_wq],axis=-1)

			# Energy network.
			l1e = tf.layers.dense(inputs=l0p_wq,units=256,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense1e")
			l1pe = tf.concat([l1e,CODES_wq],axis=-1)
			l2e = tf.layers.dense(inputs=l1pe,units=256,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense2e")
			# in the final layer use the atom code information.
			l2pe = tf.concat([l2e,CODES_wq],axis=-1)
			l3e = tf.layers.dense(l2pe,units=1,activation=None,use_bias=False,name="Dense3e")*msk
			AtomEnergies = tf.reshape(l3e,(self.batch_size,self.MaxNAtom,1))*AtomEStds+AvEs
		return AtomEnergies, AtomCharges

	def CanChargeEmbeddedModel(self, cdxyzs, weights, Zs_, zjs_, gather_inds_, pair_mask_, gauss_params, atom_codes, l_max):
		"""
		This version creates a network to integrate weight information.
		and then works like any ordinary network.
		NOTE: This network is universal in the sense that it works on ANY atom!

		Args:
			dxyzs: (nmol X maxnatom X maxneigh x 3) difference vector.
			Zs: mol X maxNatom X 1 atomic number tensor.
			zjs: nmol X maxnatom x maxneigh X 1 neighbor atomic numbers.
			gather_inds: neighbor indices.
			pair_mask: (nmol X maxnatom X maxneigh x 1) multiplicative mask.
			gauss_params: (nrad X 2) tensor of gaussian paramters.  (ang.)
			l_max: max angular momentum of embedding.
		Returns:
			mol X maxNatom X 1 tensor of atom energies, charges.
			these include constant shifts.
		"""
		eff_batch_size = self.ncan * self.batch_size
		ncase = eff_batch_size*self.MaxNAtom
		nchan = self.AtomCodes.shape[1]

		# Tile and evaluate
		dxyzs = tf.reshape(cdxyzs,(eff_batch_size,self.MaxNAtom,self.MaxNeigh_NN,3))
		Zs = tf.reshape(tf.tile(Zs_[tf.newaxis,...],[self.ncan,1,1,1]),(eff_batch_size,self.MaxNAtom,1))
		zjs = tf.reshape(tf.tile(zjs_[tf.newaxis,...],[self.ncan,1,1,1,1]),(eff_batch_size,self.MaxNAtom,self.MaxNeigh_NN,1))
		gather_inds = tf.reshape(tf.tile(gather_inds_[tf.newaxis,...],[self.ncan,1,1,1,1]),(eff_batch_size,self.MaxNAtom,self.MaxNeigh_NN,2))
		pair_mask = tf.reshape(tf.tile(pair_mask_[tf.newaxis,...],[self.ncan,1,1,1,1]),(eff_batch_size,self.MaxNAtom,self.MaxNeigh_NN,1))

		Zrs = tf.cast(tf.reshape(Zs,(ncase,-1)),self.prec)
		Atom12Real = tf.not_equal(pair_mask,0.)
		Atom12Real4 = tf.tile(Atom12Real,[1,1,1,nchan])
		Atom12Real5 = tf.tile(Atom12Real,[1,1,1,nchan+1])

		with tf.variable_scope("AtomVariance", reuse=tf.AUTO_REUSE):
			stdinit = tf.constant(np.ones(PARAMS["MAX_ATOMIC_NUMBER"]),dtype=self.prec)
			self.AtomEStd = tf.get_variable(name="AtomEStd",dtype=self.prec,initializer=stdinit)
			AtomEStds = tf.reshape(tf.gather(self.AtomEStd, Zs, axis=0),(eff_batch_size,self.MaxNAtom,1))

		jcodes0 = tf.reshape(tf.gather(atom_codes,zjs),(eff_batch_size,self.MaxNAtom,self.MaxNeigh_NN,nchan))
		jcodes = tf.where(Atom12Real4 , jcodes0 , tf.zeros_like(jcodes0))# mol X maxNatom X maxnieh X 4

		# construct embedding.
		dist_tensor = tf.clip_by_value(tf.norm(dxyzs+1.e-36,axis=-1),1e-36,1e36)
		# NMOL X MAXNATOM X MAXNATOM X NSH
		SH = tf_spherical_harmonics(dxyzs, dist_tensor, l_max)*pair_mask # mol X maxNatom X maxNeigh X nang.
		RAD = tf_gauss(dist_tensor, gauss_params)*pair_mask # mol X maxNatom X maxNeigh X nrad.
		# Perform each of the contractions.
		SHRAD = tf.einsum('mijk,mijl->mijkl',SH,RAD) # mol X maxnatom X maxneigh X nang X nrad
		emb = tf.einsum('mijkl,mijn->mikln',SHRAD,jcodes)

		# Get codes and averages of atom i
		AvEs = tf.reshape(tf.gather(self.AvE_tf, Zs, axis=0),(eff_batch_size,self.MaxNAtom,1)) # (mol * maxNatom) X 1
		AvQs = tf.reshape(tf.gather(self.AvQ_tf, Zs, axis=0),(eff_batch_size,self.MaxNAtom)) # (mol * maxNatom) X 1
		CODES = tf.reshape(tf.gather(self.atom_codes, Zs, axis=0),(ncase,nchan)) # (mol * maxNatom) X 4

		# the idea for the new network is to keep the code dimension
		# for at least the first layers.

		# Combine the codes of the main atom and the sensed atom
		# Using a hinton-esque tensor decomposition.
		with tf.variable_scope("chargenet", reuse=tf.AUTO_REUSE):
			CODEKERN1 = tf.get_variable(name="CodeKernel", shape=(nchan,nchan),dtype=self.prec)
			CODEKERN2 = tf.get_variable(name="CodeKernel2", shape=(nchan,nchan),dtype=self.prec)
			# combine the weight kernel with the codes.
			mix1 = tf.matmul(CODES,CODEKERN1) # ncase X ncode
			embrs = tf.reshape(emb,(ncase,-1,nchan))
			# Ensure any zero cases don't contribute.
			msk = tf.where(tf.equal(Zrs,0.0),tf.zeros_like(Zrs),tf.ones_like(Zrs))
			embrs *= msk[:,:,tf.newaxis]
			weighted = tf.einsum('ikj,ij->ikj',embrs,mix1)
			weighted2 = tf.einsum('ikj,jl->ikl',weighted,CODEKERN2)
			# Now pass it through as usual.
			l0 = tf.reshape(weighted2,(ncase,-1))
			l0p = tf.concat([l0,CODES],axis=-1)

			l1q = tf.layers.dense(inputs=l0p,units=512,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense1q")
			l1pq = tf.concat([l1q,CODES],axis=-1)
			l2q = tf.layers.dense(inputs=l1pq,units=512,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense2q")
			l2pq = tf.concat([l2q,CODES],axis=-1)
			l3q = tf.layers.dense(l2pq,units=1,activation=None,use_bias=False,name="Dense3q")*msk
			charges = tf.reshape(l3q,(eff_batch_size,self.MaxNAtom))
			# Set the total charges to neutral by evenly distributing any excess charge.
			excess_charges = tf.reduce_sum(charges,axis=[1])
			n_atoms = tf.reduce_sum(tf.where(tf.equal(Zs,0),Zs,tf.ones_like(Zs)),axis=[1,2])
			fix = -1.0*excess_charges/tf.cast(n_atoms,self.prec)
			AtomCharges = charges + fix[:,tf.newaxis] + AvQs

		# Now concatenate the charges onto the embedding for the energy network.
		with tf.variable_scope("energynet", reuse=tf.AUTO_REUSE):
			qcodes = tf.reshape(tf.gather_nd(AtomCharges, gather_inds),(eff_batch_size,self.MaxNAtom,self.MaxNeigh_NN,1))
			jcodes0_wq = tf.concat([jcodes,qcodes],axis=-1)
			jcodes_wq = tf.where(Atom12Real5 , jcodes0_wq , tf.zeros_like(jcodes0_wq))# mol X maxNatom X maxnieh X 4
			emb_wq = tf.einsum('mijkl,mijn->mikln',SHRAD,jcodes_wq)
			CODES_wq = tf.concat([CODES,tf.reshape(AtomCharges,(ncase,1))],axis=-1)
			CODEKERN1_wq = tf.get_variable(name="CodeKernel_wq", shape=(nchan+1,nchan+1),dtype=self.prec)
			CODEKERN2_wq = tf.get_variable(name="CodeKernel2_wq", shape=(nchan+1,nchan+1),dtype=self.prec)
			# combine the weight kernel with the codes.
			mix1_wq = tf.matmul(CODES_wq,CODEKERN1_wq) # ncase X ncode
			embrs_wq = tf.reshape(emb_wq,(ncase,-1,nchan+1))
			# Ensure any zero cases don't contribute.
			msk_wq = tf.where(tf.equal(Zrs,0.0),tf.zeros_like(Zrs),tf.ones_like(Zrs))
			embrs_wq *= msk_wq[:,:,tf.newaxis]
			weighted_wq = tf.einsum('ikj,ij->ikj',embrs_wq,mix1_wq)
			weighted2_wq = tf.einsum('ikj,jl->ikl',weighted_wq,CODEKERN2_wq)
			# Now pass it through as usual.
			l0_wq = tf.reshape(weighted2_wq,(ncase,-1))
			l0p_wq = tf.concat([l0_wq,CODES_wq],axis=-1)

			# Energy network.
			l1e = tf.layers.dense(inputs=l0p_wq,units=256,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense1e")
			l1pe = tf.concat([l1e,CODES_wq],axis=-1)
			l2e = tf.layers.dense(inputs=l1pe,units=256,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense2e")
			# in the final layer use the atom code information.
			l2pe = tf.concat([l2e,CODES_wq],axis=-1)
			l3e = tf.layers.dense(l2pe,units=1,activation=None,use_bias=False,name="Dense3e")*msk
			AtomEnergies = tf.reshape(l3e,(eff_batch_size,self.MaxNAtom,1))*AtomEStds+AvEs

		# Now recombine the predictions with the weights.
		energy_shape = (self.ncan,self.batch_size,self.MaxNAtom,1)
		charge_shape = (self.ncan,self.batch_size,self.MaxNAtom)
		eweights = tf.reshape(weights,energy_shape)
		qweights = tf.reshape(weights,charge_shape)

		self.CAEs = tf.reshape(AtomEnergies,energy_shape)
		self.CAQs = tf.reshape(AtomCharges,charge_shape)

		wCAEs = tf.where(tf.greater(eweights,1e-13),self.CAEs,tf.zeros_like(self.CAEs))
		wCAQs = tf.where(tf.greater(qweights,1e-13),self.CAQs,tf.zeros_like(self.CAQs))

		self.AtomNetEnergies = tf.reduce_sum(eweights*wCAEs,axis=0)
		self.AtomCharges = tf.reduce_sum(qweights*wCAQs,axis=0)

		# Variances of the above.
		# Perhaps to add to a loss function.
		self.AtomNetEnergies_var = tf.reduce_sum(wCAEs*wCAEs*eweights,axis=0)
		self.AtomCharges_var = tf.reduce_sum(wCAQs*wCAQs*qweights,axis=0)
		self.AtomNetEnergies_var -= self.AtomNetEnergies*self.AtomNetEnergies
		self.AtomCharges_var -= self.AtomCharges*self.AtomCharges

		return self.AtomNetEnergies, self.AtomCharges

	def train_step(self,step):
		feed_dict, mols = self.NextBatch(self.mset)
		_ , train_loss = self.sess.run([self.train_op, self.Tloss], feed_dict=feed_dict)
		if (np.isnan(train_loss)):
			print("Problem Batch discovered.")
			for m in mols:
				print(m)
		self.print_training(step, train_loss)
		return

	def print_training(self, step, loss_):
		if (step%int(500/self.batch_size)==0):
			if (not np.isnan(loss_)):
				self.saver.save(self.sess, './networks/SparseCodedGauSH', global_step=step)
			else:
				print("Reload Induced by Nan....")
				self.Load(load_averages = True)
				return
			print("step: ", "%7d"%step, "  train loss: ", "%.10f"%(float(loss_)))
			if (self.DoCodeLearning):
				print("Gauss Params: ",self.sess.run([self.gp_tf])[0])
				print("AtomCodes: ",self.sess.run([self.atom_codes])[0])
			feed_dict, mols = self.NextBatch(self.mset)
			if (self.DoChargeLearning or self.DoDipoleLearning):
				ens, frcs, charges, dipoles, qens, summary = self.sess.run([self.MolEnergies,self.MolGrads,self.AtomCharges,self.MolDipoles,self.MolCoulEnergies,self.summary_op], feed_dict=feed_dict, options=self.options, run_metadata=self.run_metadata)
			else:
				ens,frcs,summary = self.sess.run([self.MolEnergies,self.MolGrads,self.summary_op], feed_dict=feed_dict, options=self.options, run_metadata=self.run_metadata)
			for i in range(6):
				print("Pred, true: ", ens[i], feed_dict[self.groundTruthE_pl][i])
				if (self.DoChargeEmbedding):
					print("MolCoulEnergy: ", qens[i])
			diffF = frcs-feed_dict[self.groundTruthG_pl]
			MAEF = np.average(np.abs(diffF))
			RMSF = np.sqrt(np.average(diffF*diffF))
			if (MAEF > 1.0 or np.any(np.isnan(MAEF))):
				# locate the problem case.
				for i,m in enumerate(mols):
					MAEFm = np.average(np.abs(frcs[i] - feed_dict[self.groundTruthG_pl][i]))
					if (MAEFm>1.0 or np.any(np.isnan(MAEFm))):
						print("--------------")
						print(m)
						print(frcs[i])
						print(feed_dict[self.groundTruthG_pl][i])
						print("--------------")
			diffE = ens-feed_dict[self.groundTruthE_pl]
			MAEE = np.average(np.abs(diffE))
			worst_mol = np.argmax(np.abs(diffE))
			print ("Worst Molecule:",mols[worst_mol])
			if (HAS_MATPLOTLIB):
				plt.title("Histogram with 'auto' bins")
				plt.hist(diffE, bins='auto')
				plt.savefig('./logs/EError.png', bbox_inches='tight')
				# Plot error as a function of total energy.
				plt.clf()
				plt.scatter(feed_dict[self.groundTruthE_pl],diffE[:,np.newaxis])
				plt.savefig('./logs/EError2.png', bbox_inches='tight')
				plt.clf()
			RMSE = np.sqrt(np.average(np.abs(ens-feed_dict[self.groundTruthE_pl])))
			print("Mean Abs Error: (Energy)", MAEE)
			print("Mean Abs Error (Force): ", MAEF)
			print("RMS Error (Energ): ", RMSE)
			print("RMS Error (Force): ", RMSF)
			if (self.DoDipoleLearning):
				print("Mean Abs Error (Dipole): ", np.average(np.abs(dipoles-feed_dict[self.groundTruthD_pl])))
			if (self.DoChargeLearning):
				print("Mean Abs Error (Charges): ", np.average(np.abs(charges-feed_dict[self.groundTruthQ_pl])))
			if (self.DoRotGrad):
				print("RotGrad:",self.sess.run([self.RotGrad], feed_dict=feed_dict))
			self.writer.add_summary(summary,step)
		return

	def training(self, loss):
		optimizer = tf.train.AdamOptimizer(learning_rate=(self.learning_rate))
		grads = tf.gradients(loss, tf.trainable_variables())
		# Avoid any nans
		for grad in grads:
			grad = tf.where(tf.is_nan(grad),tf.zeros_like(grad),grad)
		vars = tf.trainable_variables()
		for var in vars:
			var = tf.where(tf.is_nan(var),tf.zeros_like(var),var)
		grads, _ = tf.clip_by_global_norm(grads, 50)
		grads_and_vars = list(zip(grads, vars))
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
		return train_op

	def Train(self,mxsteps=500000):
		test_freq = 40
		for step in range(1, mxsteps+1):
			self.train_step(step)
		return

	@TMTiming("Prepare")
	def Prepare(self):
		tf.reset_default_graph()
		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		self.DoRotGrad = False
		self.DoForceLearning = True
		if(self.Lsq):
			self.Canonicalize = False
		else:
			self.Canonicalize = True
		self.DoCodeLearning = True
		self.DoDipoleLearning = False
		self.DoChargeLearning = True
		self.DoChargeEmbedding = True

		if (self.mode == 'eval'):
			self.DoForceLearning = False
			self.DoCodeLearning = False
			self.DoDipoleLearning = False
			self.DoChargeLearning = False

		self.xyzs_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,3), dtype = self.prec, name="InputCoords")
		self.zs_pl = tf.placeholder(shape = (self.batch_size, self.MaxNAtom, 1), dtype = tf.int32, name="InputZs")
		self.nl_nn_pl = tf.placeholder(shape = (self.batch_size, self.MaxNAtom, self.MaxNeigh_NN), dtype = tf.int32,name="InputNL_NN")
		self.nl_j_pl = tf.placeholder(shape = (self.batch_size, self.MaxNAtom, self.MaxNeigh_J), dtype = tf.int32,name="InputNL_J")

		self.cax_pl = tf.placeholder(shape = (self.ncan, self.batch_size*self.MaxNAtom, 3,3), dtype = self.prec, name="CanAxes")
		self.cw_pl = tf.placeholder(shape = (self.ncan, self.batch_size, self.MaxNAtom), dtype = self.prec, name="CanWeights")

		# Learning targets.
		self.groundTruthE_pl = tf.placeholder(shape = (self.batch_size,1), dtype = self.prec,name="GTEs") # Energies
		self.groundTruthG_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,3), dtype = self.prec,name="GTFs") # Forces
		self.groundTruthD_pl = tf.placeholder(shape = (self.batch_size,3), dtype = self.prec,name="GTDs") # Dipoles.
		self.groundTruthQ_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom), dtype = self.prec,name="GTQs") # Charges

		# Constants
		self.atom_codes = tf.Variable(self.AtomCodes,trainable=self.DoCodeLearning,dtype = self.prec,name="atom_codes")
		self.gp_tf  = tf.Variable(self.GaussParams,trainable=self.DoCodeLearning, dtype = self.prec,name="gauss_params")
		self.AvE_tf = tf.Variable(self.AverageElementEnergy, trainable=False, dtype = self.prec,name="av_energies")
		self.AvQ_tf = tf.Variable(self.AverageElementCharge, trainable=False, dtype = self.prec,name="av_charges")

		self.MaxNeigh_NN_prep = self.MaxNeigh_NN
		self.MaxNeigh_J_prep = self.MaxNeigh_J

		Zg0 = tf.greater(self.zs_pl,0)
		NAtomsPerMol = tf.reduce_sum(tf.cast(Zg0,self.prec),axis=1,keepdims=True)
		Atom1Real = tf.tile(Zg0[:,:,tf.newaxis,:],(1,1,self.MaxNeigh_NN,1))
		nl = tf.reshape(self.nl_nn_pl,(self.batch_size,self.MaxNAtom,self.MaxNeigh_NN,1))
		Atom12Real = tf.logical_and(Atom1Real,tf.greater_equal(nl,0))
		Atom12Real2 = tf.tile(Atom12Real,[1,1,1,2])
		Atom12Real3 = tf.tile(Atom12Real,[1,1,1,3])
		Atom12Real4 = tf.tile(Atom12Real,[1,1,1,4])

		Atom1Real_j = tf.tile(tf.greater(self.zs_pl,0)[:,:,tf.newaxis,:],(1,1,self.MaxNeigh_J,1))
		nl_j = tf.reshape(self.nl_j_pl,(self.batch_size,self.MaxNAtom,self.MaxNeigh_J,1))
		Atom12Real_j = tf.logical_and(Atom1Real_j,tf.greater_equal(nl_j,0))
		Atom12Real2_j = tf.tile(Atom12Real_j,[1,1,1,2])
		Atom12Real3_j = tf.tile(Atom12Real_j,[1,1,1,3])

		molis = tf.tile(tf.range(self.batch_size)[:,tf.newaxis,tf.newaxis],[1,self.MaxNAtom,self.MaxNeigh_NN])[:,:,:,tf.newaxis]
		gather_inds0 = tf.concat([molis,nl],axis=-1)
		it1 = (self.MaxNAtom-1)*tf.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_NN,1),dtype=tf.int32)
		gather_inds0p = tf.concat([molis,it1],axis=-1)
		gather_inds = tf.where(Atom12Real2, gather_inds0, gather_inds0p) # Mol X MaxNatom X maxN X 2
		self.sparse_mask = tf.cast(Atom12Real,self.prec) # nmol X maxnatom X maxneigh X 1

		molis_j = tf.tile(tf.range(self.batch_size)[:,tf.newaxis,tf.newaxis],[1,self.MaxNAtom,self.MaxNeigh_J])[:,:,:,tf.newaxis]
		gather_inds0_j = tf.concat([molis_j,nl_j],axis=-1)
		it1_j = (self.MaxNAtom-1)*tf.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_J,1),dtype=tf.int32)
		gather_inds0p_j = tf.concat([molis_j,it1_j],axis=-1)
		gather_inds_j = tf.where(Atom12Real2_j, gather_inds0_j, gather_inds0p_j) # Mol X MaxNatom X maxN X 2

		# sparse version of dxyzs.
		if self.DoRotGrad:
			thetas = tf.acos(2.0*tf.random_uniform([self.batch_size],dtype=self.prec)-1.0)
			phis = tf.random_uniform([self.batch_size],dtype=self.prec)*2*Pi
			psis = tf.random_uniform([self.batch_size],dtype=self.prec)*2*Pi
			matrices = TF_RotationBatch(thetas,phis,psis)
			xyzs_shifted = self.xyzs_pl - self.xyzs_pl[:,0,:][:,tf.newaxis,:]
			tmpxyzs = tf.einsum('ijk,ikl->ijl',xyzs_shifted, matrices)
		else:
			tmpxyzs = self.xyzs_pl

		zjs0 = tf.gather_nd(self.zs_pl, gather_inds) # mol X maxNatom X maxNeigh X 1
		zjs = tf.where(Atom12Real, zjs0, tf.zeros_like(zjs0)) # mol X maxNatom X maxneigh X 1
		coord1 = tf.expand_dims(tmpxyzs, axis=2) # mol X maxnatom X 1 X 3
		coord2 = tf.gather_nd(tmpxyzs,gather_inds)
		diff0 = (coord1-coord2)
		self.dxyzs = tf.where(Atom12Real3, diff0, tf.zeros_like(diff0))

		self.AtomNetEnergies = tf.zeros((self.batch_size,self.MaxNAtom,1),dtype=self.prec,name='AtomEnergies')
		self.AtomCharges = tf.zeros((self.batch_size,self.MaxNAtom),dtype=self.prec,name='AtomCharges')

		if (self.Canonicalize):
			if 1:#(self.mode=='train'):
				weights,axs = self.CanonicalAxes(self.dxyzs,self.sparse_mask)
			else:
				weights,axs = self.cw_pl, self.cax_pl
			rds = tf.reshape(self.dxyzs,(self.batch_size*self.MaxNAtom,self.MaxNeigh_NN,3))
			cdxyzs = tf.reshape(tf.einsum('ijk,milk->mijl',rds,axs),(self.ncan,self.batch_size,self.MaxNAtom,self.MaxNeigh_NN,3))
			self.AtomNetEnergies,self.AtomCharges = self.CanChargeEmbeddedModel(cdxyzs, weights, self.zs_pl, zjs, gather_inds, self.sparse_mask, self.gp_tf, self.atom_codes, self.l_max)
			self.EVarianceLoss = tf.nn.l2_loss(self.AtomNetEnergies_var)
			tf.summary.scalar('EVarLoss',self.EVarianceLoss)
			tf.add_to_collection('EVarLoss', self.EVarianceLoss)
		else:
			self.AtomNetEnergies,self.AtomCharges = self.ChargeEmbeddedModel(self.dxyzs, self.zs_pl, zjs, gather_inds, self.sparse_mask, self.gp_tf, self.atom_codes, self.l_max)

		if (self.DoChargeLearning or self.DoDipoleLearning):
			self.MolDipoles = self.ChargeToDipole(self.xyzs_pl,self.zs_pl,self.AtomCharges)
			self.Qloss = tf.nn.l2_loss(self.AtomCharges - self.groundTruthQ_pl,name='Qloss')/tf.cast(self.batch_size*self.MaxNAtom,self.prec)
			self.Dloss = tf.nn.l2_loss(self.MolDipoles - self.groundTruthD_pl,name='Dloss')/tf.cast(self.batch_size,self.prec)
			tf.summary.scalar('Qloss',self.Qloss)
			tf.summary.scalar('Dloss',self.Dloss)
			tf.add_to_collection('losses', self.Qloss)
			tf.add_to_collection('losses', self.Dloss)
		if (self.DoChargeEmbedding):
			coord2_j = tf.gather_nd(tmpxyzs,gather_inds_j)
			diff0_j = (coord1-coord2_j)
			dxyzs_j = tf.where(Atom12Real3_j, diff0_j, tf.zeros_like(diff0_j))
			q2 = tf.gather_nd(self.AtomCharges, gather_inds_j)
			q1q2unmsk = (self.AtomCharges[:,:,tf.newaxis]*q2)
			q1q2s = tf.where(Atom12Real_j[:,:,:,0],q1q2unmsk,tf.zeros_like(q1q2unmsk))
			self.AtomCoulEnergies = tf.where(tf.greater(self.zs_pl,0),self.CoulombAtomEnergies(tf.norm(dxyzs_j,axis=-1),q1q2s),tf.zeros_like(self.AtomNetEnergies))
		else:
			self.AtomCoulEnergies = tf.zeros_like(self.AtomNetEnergies)

		self.MolCoulEnergies = tf.reduce_sum(self.AtomCoulEnergies,axis=1,keepdims=False)

		if (self.DoChargeEmbedding):
			self.AtomEnergies = self.AtomNetEnergies + self.AtomCoulEnergies
		else:
			self.AtomEnergies = self.AtomNetEnergies
		self.MolEnergies = tf.reduce_sum(self.AtomEnergies,axis=1,keepdims=False)

		# Optional. Verify that the canonicalized differences are invariant.
		if self.DoRotGrad:
			self.RotGrad = tf.gradients(self.AtomNetEnergies,psis)[0]
			tf.summary.scalar('RotGrad',tf.reduce_sum(self.RotGrad))

		MolGradsRaw = tf.gradients(self.MolEnergies,self.xyzs_pl)[0]
		msk = tf.tile(tf.not_equal(self.zs_pl,0),[1,1,3])
		self.MolGrads = tf.where(msk,MolGradsRaw,tf.zeros_like(MolGradsRaw))

		if (self.DoHess):
			self.MolHess = tf.hessians(self.MolEnergies,self.xyzs_pl)[0]

		if (self.mode=='train'):
			self.Eloss = tf.nn.l2_loss((self.MolEnergies - self.groundTruthE_pl)*NAtomsPerMol,name='Eloss')/tf.cast(self.batch_size,self.prec)
			tf.add_to_collection('losses', self.Eloss)
			tf.summary.scalar('ELoss',self.Eloss)
			self.Tloss = self.Eloss

			t1 = tf.reshape(self.MolGrads,(self.batch_size,-1))
			t2 = tf.reshape(self.groundTruthG_pl,(self.batch_size,-1))
			diff = t1 - t2
			self.GradDiff = tf.clip_by_value(diff*diff,1e-36,1.0)
			self.Gloss = tf.reduce_sum(self.GradDiff)
			tf.losses.add_loss(self.Gloss,loss_collection=tf.GraphKeys.LOSSES)
			tf.summary.scalar('Gloss',self.Gloss)
			tf.add_to_collection('losses', self.Gloss)

			if (self.Canonicalize):
				CanTotalAtEns = self.CAEs + self.AtomCoulEnergies[tf.newaxis,...]
				CanEns = tf.reduce_sum(CanTotalAtEns,axis=2,keepdims=False)
				self.CEloss = tf.nn.l2_loss(CanEns - self.groundTruthE_pl[tf.newaxis,...],name='CEloss')/tf.cast(self.batch_size*self.ncan,self.prec)
				tf.summary.scalar('CELoss',self.CEloss)
				self.Tloss += self.CEloss

				if (self.DoForceLearning):
					LCanEns = tf.unstack(CanEns,axis=0)
					CanGrads0 = tf.stack([tf.gradients(XXX,self.xyzs_pl)[0] for XXX in LCanEns],axis=0)
					cmsk = tf.tile(msk[tf.newaxis,...],[self.ncan,1,1,1])
					CanGrads = tf.where(cmsk,CanGrads0,tf.zeros_like(CanGrads0))
					cdiff = tf.reshape(CanGrads,(self.ncan,self.batch_size,-1)) - t2[tf.newaxis,...]
					self.CGradDiff = tf.clip_by_value(cdiff*cdiff,1e-36,1.0)
					self.CGloss = tf.reduce_sum(self.GradDiff)/tf.cast(self.batch_size*self.MaxNAtom*3*self.ncan,self.prec)
					tf.losses.add_loss(self.CGloss,loss_collection=tf.GraphKeys.LOSSES)
					tf.summary.scalar('CGloss',self.CGloss)
					self.Tloss += 5.*self.CGloss

			if (self.DoForceLearning):
				self.Tloss += (self.Gloss)

			if (self.DoDipoleLearning):
				self.Tloss += (self.Dloss)
			elif (self.DoChargeLearning):
				self.Tloss += (self.Qloss)

			tf.losses.add_loss(self.Tloss,loss_collection=tf.GraphKeys.LOSSES)
			tf.summary.scalar('TLoss',self.Tloss)
			tf.add_to_collection('losses', self.Tloss)
			self.train_op = self.training(self.Tloss)

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		self.writer = tf.summary.FileWriter('./networks/SparseCodedGauSH', graph=tf.get_default_graph())
		self.summary_op = tf.summary.merge_all()

		if (False):
			print("logging with FULL TRACE")
			self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			self.run_metadata = tf.RunMetadata()
			self.writer.add_run_metadata(self.run_metadata, "init", global_step=None)
		else:
			self.options = None
			self.run_metadata = None
		self.sess.run(self.init)
		#self.sess.graph.finalize()

#net = SparseCodedChargedGauSHNetwork(aset=b,load=True,load_averages=True,mode='train')
#net.Train()
#net = SparseCodedChargedGauSHNetwork(aset=None,load=True,load_averages=True,mode='eval')
