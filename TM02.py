"""
This revision shares low-layers to avoid overfitting, and save time.
It's a little recurrent-y.
"""

from TensorMol import *
import numpy as np

HAS_MATPLOTLIB=True
if (HAS_MATPLOTLIB):
	import matplotlib.pyplot as plt
if (0):
	# Linear molecules are a good test.
	m=Mol()
	m.FromXYZString("""5
	Comment:
	C   0.1506907344  0.0  0.0
	C   -1.1550766934  0.0  0.0
	S   -2.7208105787  0.0  0.0
	C   1.4245508222  0.0  0.0
	S   2.9488539653  0.0  0.0""")
	m.properties["energy"]=-1000.0
	m.properties["charges"]=np.array([0.,0.,0.,0.,0.])
	m.properties["gradients"]=np.zeros((5,3))
	m.properties["dipole"]=np.zeros((3))
	for i in range(300):
		b.mols.append(m)

if (0):
	a = MSet("chemspider20_1_opt_withcharge_noerror_all")
	b = MSet("chemspider20_1_opt_withcharge_noerror_part2_max50")
	c = MSet("chemspider12_clean_maxatom35")
	d = MSet("kevin_heteroatom.dat")
	a.Load()
	b.Load()
	c.Load()
	d.Load()
	b.mols = a.mols+b.mols+c.mols[:len(b.mols)]+d.mols
	#b.Statistics()
	b.cut_max_num_atoms(55)
	b.cut_max_grad(2.0)
	b.Save("Hybrid2")

if 0:
	a = MSet("kevin_rand1")
	b = MSet("chemspider20_1_meta_withcharge_noerror_all")
	c = MSet("kevin_heteroatom.dat")
	d = MSet("chemspider12_clean_maxatom35_small")
	e = MSet("chemspider20_1_opt_withcharge_noerror_all")
	a.Load()
	b.Load()
	c.Load()
	d.Load()
	d.cut_randomselection(300000)
	e.Load()
	b.mols = a.mols+b.mols+c.mols+d.mols+e.mols
	b.cut_max_num_atoms(40)
	b.cut_max_grad(2.0)
	b.cut_energy_outliers()
	b.Save("Hybrid")

if 0:
	#b = MSet("chemspider20_1_meta_withcharge_noerror_all")
	b = MSet("Hybrid")
	b.Load()
	b.cut_max_num_atoms(55)
	b.cut_max_grad(1.0)
	b.cut_energy_outliers(0.5)

MAX_ATOMIC_NUMBER = 55

def sftpluswparam(x):
	return tf.log(1.0+tf.exp(100.*x))/100.0

def safe_inv_norm(x_):
	nrm = tf.clip_by_value(tf.norm(x_,axis=-1,keepdims=True),1e-36,1e36)
	nrm_ok = tf.logical_and(tf.not_equal(nrm,0.),tf.logical_not(tf.is_nan(nrm)))
	safe_nrm = tf.where(nrm_ok,nrm,tf.ones_like(nrm))
	return tf.where(nrm_ok,1.0/safe_nrm,tf.zeros_like(nrm))

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

def RemoveReflections(dxyzs):
	"""
	If dxyzs is going to be used in a way which is invariant to reflections.
	this limits the space of possible dxyzs by ensuring only 1/octant
	of distances is available. (those with positive averages.)
	Ie: resulting energies are invariant WRT inversion & reflections.
	"""
	avx = tf.reduce_sum(dxyzs[:,:,:,:1],axis=2,keepdims=True)
	ot = tf.ones_like(avx)
	signsx = tf.where(tf.less_equal(avx,0.),-1.*ot,ot)
	signsy = tf.where(tf.less_equal(tf.reduce_sum(dxyzs[:,:,:,1:2],axis=2,keepdims=True),0.),-1.*ot,ot)
	signsz = tf.where(tf.less_equal(tf.reduce_sum(dxyzs[:,:,:,2:],axis=2,keepdims=True),0.),-1.*ot,ot)
	signtensor = tf.concat([signsx,signsy,signsz],axis=-1)
	return dxyzs * signtensor

def CanonicalizeGS(dxyzs,z2s):
	"""
	This version returns two sets of axes for nearest and next-nearest neighbor.
	If the energy from both these representations is averaged the result
	will be permutationally invariant (WRT nearest-next-nearest motion)
	and rotationally invariant.

	Args:
		dxyz: a nMol X maxNatom X maxNatom X 3 tensor of atoms. (differenced from center of embedding
		zs: a nMol X maxNatom X maxNatom X 1 tensor of atomic number pairs.
		ie: ... X i X i = (0.,0.,0.))
	"""
	# Append orthogonal axes to dxyzs
	argshape = tf.shape(dxyzs)
	realdata = tf.reshape(dxyzs,(argshape[0]*argshape[1],argshape[2],3))

	v1 = tf.reshape(dxyzs[:,:,:1,:],(argshape[0]*argshape[1],3))+tf.constant(np.array([1e-6,0.,0.]),dtype=tf.float64)
	v1 *= safe_inv_norm(v1)
	v2 = tf.reshape(dxyzs[:,:,1:2,:],(argshape[0]*argshape[1],3))+tf.constant(np.array([0.,0.1e-6,0.]),dtype=tf.float64)
	v2 -= tf.einsum('ij,ij->i',v1,v2)[:,tf.newaxis]*v1
	v2 *= safe_inv_norm(v2)
	v3 = tf.cross(v1,v2)
	v3 *= safe_inv_norm(v3)

	v1p = tf.reshape(dxyzs[:,:,1:2,:],(argshape[0]*argshape[1],3))+tf.constant(np.array([0.,0.1e-6,0.]),dtype=tf.float64)
	v1p *= safe_inv_norm(v1p)
	v2p = tf.reshape(dxyzs[:,:,:1,:],(argshape[0]*argshape[1],3))+tf.constant(np.array([1e-6,0.,0.]),dtype=tf.float64)
	v2p -= tf.einsum('ij,ij->i',v1p,v2p)[:,tf.newaxis]*v1p
	v2p *= safe_inv_norm(v2p)
	v3p = tf.cross(v1p,v2p)
	v3p *= safe_inv_norm(v3p)

	vs = tf.concat([v1[:,tf.newaxis,:],v2[:,tf.newaxis,:],v3[:,tf.newaxis,:]],axis=1)
	vsp = tf.concat([v1p[:,tf.newaxis,:],v2p[:,tf.newaxis,:],v3p[:,tf.newaxis,:]],axis=1)
	tore = tf.einsum('ijk,ilk->ijl',realdata,vs)
	torep = tf.einsum('ijk,ilk->ijl',realdata,vsp)
	return tf.reshape(tore,tf.shape(dxyzs)),tf.reshape(torep,tf.shape(dxyzs)),vs

class SparseCodedChargedGauSHNetwork:
	"""
	This is the basic TensorMol0.2 model chemistry.
	"""
	def __init__(self,aset=None,load=False):
		self.prec = tf.float64
		self.batch_size = 64 # Force learning strongly modulates what you can do.
		self.MaxNAtom = 32
		self.MaxNeigh = self.MaxNAtom
		self.learning_rate = 0.00003
		self.RCut = 15.0
		self.MinMAE = 100.0 # used to only save best results.
		self.AtomCodes = ELEMENTCODES
		#self.AtomCodes = np.random.random(size=(MAX_ATOMIC_NUMBER,6))
		self.AtomTypes = [1,6,7,8]
		self.l_max = 3
		self.GaussParams = np.array([[ 0.38664542,  0.26217287], [ 0.67811722,  0.23477701],    [ 1.04543342,  0.23426948],    [ 1.38311757,  0.21758330],    [ 1.68369538,  0.21645779],    [ 2.04304538,  0.21420768],    [ 2.78418335,  0.15554105],    [ 3.13734002,  0.18086331],    [ 3.79258319,  0.17154482],    [ 4.90203694,  0.11153887],    [ 5.50218806,  0.10848024]])
		self.nrad = len(self.GaussParams)
		self.nang = (self.l_max+1)**2
		self.ncodes = self.AtomCodes.shape[-1]
		self.ngaush = self.nrad*self.nang
		self.nembdim = self.ngaush*self.ncodes
		self.mset = aset
		if (aset != None):
			self.MaxNAtom = b.MaxNAtom()+1
			self.AtomTypes = b.AtomTypes()
			print("Calculating element averages...")
			AvE,AvQ = aset.RemoveElementAverages()
			self.AverageElementEnergy = np.zeros((MAX_ATOMIC_NUMBER))
			self.AverageElementCharge = np.zeros((MAX_ATOMIC_NUMBER))
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
			NLT, MaxNeigh = self.NLTensors(xyzs,zs)
			# The maximum number of non-negative indices in dimension 2
			self.MaxNeigh = MaxNeigh
			print("self.MaxNeigh = ", self.MaxNeigh)
			# Check this calculation...
			if 0:
				tmp = np.where(NLT>=0,np.ones_like(NLT),np.zeros_like(NLT))
				for i in range(50):
					print("---------------")
					print("Num At,Neigh",i,np.sum(tmp[i,0]),np.sum(tmp[i],axis=1))
					print(aset.mols[mol_order[i]])
					print(NLT[i].shape,NLT[i])
		self.sess = None
		if (load):
			self.Load()
		self.Prepare()
		return

	def Load(self):
		try:
			chkpt = tf.train.latest_checkpoint('./networks/')
			metafilename = ".".join([chkpt, "meta"])
			if (self.sess == None):
				self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			print("Loading Graph: ",metafilename)
			saver = tf.train.import_meta_graph(metafilename)
			saver.restore(self.sess,chkpt)
			print("Loaded Check: ",chkpt)
			# Recover key variables.
			self.AtomCodes = self.sess.run([v for v in tf.global_variables() if v.name == "atom_codes:0"][0])
			self.GaussParams = self.sess.run([v for v in tf.global_variables() if v.name == "gauss_params:0"][0])
			self.AverageElementEnergy = self.sess.run([v for v in tf.global_variables() if v.name == "av_energies:0"][0])
			self.AverageElementCharge = self.sess.run([v for v in tf.global_variables() if v.name == "av_charges:0"][0])
		except Exception as Ex:
			print("Load failed.",Ex)
			raise Ex
		return

	def GetDebugRoutine(self,m):
		if (m.NAtoms() > self.MaxNAtom):
			self.MaxNAtom = m.NAtoms()
		self.batch_size=1
		self.Prepare()
		self.Load()
		def EF(xyz_):
			xyzs = np.zeros((self.batch_size,self.MaxNAtom,3))
			Zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
			nls = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh),dtype=np.int32)
			xyzs[0,:m.NAtoms(),:] = xyz_
			Zs[0,:m.NAtoms(),0] = m.atoms
			nlt,MaxNeigh = self.NLTensors(xyzs,Zs)
			if (MaxNeigh > self.MaxNeigh):
				print("Too Many Neighbors.")
				raise Exception('NeighborOverflow')
			nls[:nlt.shape[0],:nlt.shape[1],:nlt.shape[2]] = nlt
			feed_dict = {self.xyzs_pl:xyzs, self.zs_pl:Zs,self.nl_pl:nls}
			print("FEED:",feed_dict)
			if 0:
				print("--- Debug Gradient ---")
				print("D-dxyzs",self.sess.run(tf.gradients(self.dxyzs,self.xyzs_pl), feed_dict=feed_dict))
				print("D-cdxyzs",self.sess.run(tf.gradients(self.cdxyzs,self.xyzs_pl), feed_dict=feed_dict))
				print("D-cdxyzs",self.sess.run(tf.gradients(self.cdxyzs,self.xyzs_pl), feed_dict=feed_dict))
				print("D-self.AtomNetEnergies",self.sess.run(tf.gradients(self.AtomNetEnergies,self.xyzs_pl), feed_dict=feed_dict))
				print("D-self.AtomNetEnergies",self.sess.run(tf.gradients(self.AtomCharges,self.xyzs_pl), feed_dict=feed_dict))
			ens,fs = self.sess.run([self.MolEnergies,self.MolGrads], feed_dict=feed_dict)
			return ens[0],fs[0][:m.NAtoms()]*(-JOULEPERHARTREE)
		return EF

	def GetEnergyForceRoutine(self,m):
		if (m.NAtoms() > self.MaxNAtom):
			self.MaxNAtom = m.NAtoms()
		self.batch_size=1
		self.Prepare()
		self.Load()
		def EF(xyz_,DoForce=True):
			xyzs = np.zeros((self.batch_size,self.MaxNAtom,3))
			Zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
			nls = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh),dtype=np.int32)
			xyzs[0,:m.NAtoms(),:] = xyz_
			Zs[0,:m.NAtoms(),0] = m.atoms
			nlt,MaxNeigh = self.NLTensors(xyzs,Zs)
			if (MaxNeigh > self.MaxNeigh):
				print("Too Many Neighbors.")
				raise Exception('NeighborOverflow')
			nls[:nlt.shape[0],:nlt.shape[1],:nlt.shape[2]] = nlt
			feed_dict = {self.xyzs_pl:xyzs, self.zs_pl:Zs,self.nl_pl:nls}
			if (self.DoRotGrad):
				print("RotGrad:",self.sess.run([self.RotGrad], feed_dict=feed_dict))
			if (DoForce):
				ens,fs = self.sess.run([self.MolEnergies,self.MolGrads], feed_dict=feed_dict)
				return ens[0],fs[0][:m.NAtoms()]*(-JOULEPERHARTREE)
			else:
				ens = self.sess.run(self.MolEnergies, feed_dict=feed_dict)[0]
				return ens[0]
		return EF

	def GetEnergyForceHessRoutine(self,m):
		if (m.NAtoms() > self.MaxNAtom):
			self.MaxNAtom = m.NAtoms()
		self.batch_size=1
		self.Prepare()
		self.Load()
		def EFH(xyz_):
			xyzs = np.zeros((self.batch_size,self.MaxNAtom,3))
			Zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
			nls = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh),dtype=np.int32)
			xyzs[0,:m.NAtoms(),:] = xyz_
			Zs[0,:m.NAtoms(),0] = m.atoms
			nlt,MaxNeigh = self.NLTensors(xyzs,Zs)
			if (MaxNeigh > self.MaxNeigh):
				print("Too Many Neighbors.")
				raise Exception('NeighborOverflow')
			nls[:nlt.shape[0],:nlt.shape[1],:nlt.shape[2]] = nlt
			feed_dict = {self.xyzs_pl:xyzs, self.zs_pl:Zs,self.nl_pl:nls}
			ens,fs,hs = self.sess.run([self.MolEnergies,self.MolGrads,self.MolHess], feed_dict=feed_dict)
			return ens[0], fs[0][:m.NAtoms()]*(-JOULEPERHARTREE), hs[0][:m.NAtoms()][:m.NAtoms()]*JOULEPERHARTREE*JOULEPERHARTREE
		return EFH

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
		nls = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh),dtype=np.int32)
		true_ae = np.zeros((self.batch_size,1),dtype=np.float)
		qs = np.zeros((self.batch_size,self.MaxNAtom),dtype=np.float64) # Charges.
		ds = np.zeros((self.batch_size,3),dtype=np.float64) # Dipoles.
		mols = []
		for i in range(self.batch_size):
			mi = np.random.randint(len(aset.mols))
			m = aset.mols[mi]
			mols.append(m)
			xyzs[i,:m.NAtoms()] = m.coords
			zs[i,:m.NAtoms(),0] = m.atoms
			true_ae[i]=m.properties["energy"]
			true_force[i,:m.NAtoms()]=m.properties["gradients"]
			qs[i,:m.NAtoms()]=m.properties["charges"]
			ds[i]=m.properties["dipole"]
		nlt, MaxNeigh = self.NLTensors(xyzs,zs)
		if (MaxNeigh > self.MaxNeigh):
			print("Too Many Neighbors.")
			raise Exception('NeighborOverflow')
		nls[:nlt.shape[0],:nlt.shape[1],:nlt.shape[2]] = nlt
		return {self.xyzs_pl:xyzs, self.zs_pl:zs,self.nl_pl:nls,self.groundTruthE_pl:true_ae, self.groundTruthG_pl:true_force, self.groundTruthQ_pl:qs, self.groundTruthD_pl:ds}, mols

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
		nlt = Make_NLTensor(xyzs_,zs_.astype(np.int32),self.RCut, self.MaxNAtom, True)
		return nlt, nlt.shape[-1]

	def CoulombAtomEnergies(self,dxyzs_,q1q2s_):
		"""
		Atom Coulomb energy with polynomial cutoff.
		Zero of energy is set so that if all atom pairs are outside the cutoff the energy is zero.

		Args:
			dxyzs_: (nmol X MaxNAtom X MaxNeigh X 1) distances tensor (angstrom)
			# NOTE: This has no self-energy because of the way dxyzs_ is neighbored.
			q1q2s_: charges (nmol X MaxNAtom X MaxNeigh X 1) charges tensor. (atomic)
		Returns:
			 (nmol X 1) tensor of energies (atomic)
		"""
		EMR = (1.0/2.0)*tf.reduce_sum(polykern(dxyzs_*1.889725989)*q1q2s_,axis=(2))[:,:,tf.newaxis]
		KLR = tf.ones_like(dxyzs_)*(1.0/self.RCut)
		ELR = (1.0/2.0)*tf.reduce_sum(KLR*q1q2s_,axis=(2))[:,:,tf.newaxis]
		return EMR-ELR

	def ChargeToDipole(self,xyzs_,zs_,qs_):
		"""
		Calculate the dipole moment relative to center of atom.
		"""
		n_atoms = tf.clip_by_value(tf.cast(tf.reduce_sum(zs_,axis=(1,2)),tf.float64),1e-36,1e36)
		COA = tf.reduce_sum(xyzs_,axis=1)/n_atoms[:,tf.newaxis]
		return tf.reduce_sum((xyzs_ - COA[:,tf.newaxis,:])*qs_[:,:,tf.newaxis],axis=1)

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

		jcodes0 = tf.reshape(tf.gather(atom_codes,zjs),(self.batch_size,self.MaxNAtom,self.MaxNeigh,nchan))
		jcodes = tf.where(Atom12Real4 , jcodes0 , tf.zeros_like(jcodes0))# mol X maxNatom X maxnieh X 4

		# construct embedding.
		dist_tensor = tf.clip_by_value(tf.norm(dxyzs+1.e-36,axis=-1),1e-36,1e36)

		# NMOL X MAXNATOM X MAXNATOM X NSH
		SH = tf_spherical_harmonics(dxyzs, dist_tensor, l_max)*pair_mask # mol X maxNatom X maxNeigh X nang.
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

			l1q = tf.layers.dense(inputs=l0p,units=256,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense1q")
			l1pq = tf.concat([l1q,CODES],axis=-1)
			l2q = tf.layers.dense(inputs=l1pq,units=256,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer,name="Dense2q")
			l2pq = tf.concat([l2q,CODES],axis=-1)
			l3q = tf.layers.dense(l2pq,units=1,activation=None,use_bias=False,name="Dense3q")*msk
			charges = tf.reshape(l3q,(self.batch_size,self.MaxNAtom))
			# Set the total charges to neutral by evenly distributing any excess charge.
			excess_charges = tf.reduce_sum(charges,axis=[1])
			n_atoms = tf.reduce_sum(tf.where(tf.equal(Zs,0),Zs,tf.ones_like(Zs)),axis=[1,2])
			fix = -1.0*excess_charges/tf.cast(n_atoms,tf.float64)
			AtomCharges = charges + fix[:,tf.newaxis] + AvQs

		# Now concatenate the charges onto the embedding for the energy network.
		with tf.variable_scope("energynet", reuse=tf.AUTO_REUSE):
			qcodes = tf.reshape(tf.gather_nd(AtomCharges, gather_inds),(self.batch_size,self.MaxNAtom,self.MaxNeigh,1))
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
			AtomEnergies = tf.reshape(l3e,(self.batch_size,self.MaxNAtom,1))+AvEs

		return AtomEnergies, AtomCharges


	def train_step(self,step):
		feed_dict, mols = self.NextBatch(self.mset)
		#DEBUGFLEARNING = self.sess.run(tf.gradients(self.Gloss,tf.trainable_variables()), feed_dict=feed_dict)[0]
		#print(DEBUGFLEARNING)
		#for t in DEBUGFLEARNING:
		#	if (np.any(np.isnan(t))):
		#		print("NanLearning!!!", t)
		#tvars = tf.trainable_variables()
		#for var in tvars:
			#print("var", var)
		if 0:
			a,b = self.sess.run([self.dxyzs, self.cdxyzs_p], feed_dict=feed_dict)
			for i,d in enumerate(a[:10]):
				print(mols[i])
				print(" --- ",d)
			for i,d in enumerate(b[:10]):
				print(mols[i])
				print(" --- ",d)
		_ , train_loss = self.sess.run([self.train_op, self.Tloss], feed_dict=feed_dict)
		self.print_training(step, train_loss)
		return

	def print_training(self, step, loss_):
		if (step%15==0):
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
				#print("Zs:",feed_dict[self.zs_pl][i])
				#print("xyz:",feed_dict[self.xyzs_pl][i])
				#print("NL:",feed_dict[self.nl_pl][i])
				#print("Pred, true: ", ens[i], feed_dict[self.groundTruthE_pl][i], frcs[i], feed_dict[self.groundTruthG_pl][i] )
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
			if (MAEE < self.MinMAE):
				print("New Min, Saving...")
				self.MinMAE = MAEE
				self.saver.save(self.sess, './networks/SparseCodedGauSH', global_step=step)
			self.writer.add_summary(summary,step)
		return

	def training(self, loss):
		optimizer = tf.train.AdamOptimizer(learning_rate=(self.learning_rate))
		grads = tf.gradients(loss, tf.trainable_variables())
		grads, _ = tf.clip_by_global_norm(grads, 50)
		grads_and_vars = list(zip(grads, tf.trainable_variables()))
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
		return train_op

	def Train(self,mxsteps=500000):
		test_freq = 40
		for step in range(1, mxsteps+1):
			self.train_step(step)
		return

	def Prepare(self):
		tf.reset_default_graph()

		self.DoRotGrad = False
		self.DoForceLearning = True
		self.Canonicalize = True
		self.DoUnChiralize = True # Enforces inversion symmetry.
		self.DoCodeLearning = False
		self.DoDipoleLearning = False
		self.DoChargeLearning = True
		self.DoChargeEmbedding = True

		self.xyzs_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,3), dtype = self.prec, name="InputCoords")
		self.zs_pl = tf.placeholder(shape = (self.batch_size, self.MaxNAtom, 1), dtype = tf.int32, name="InputZs")
		self.nl_pl = tf.placeholder(shape = (self.batch_size, self.MaxNAtom, self.MaxNeigh), dtype = tf.int32,name="InputNL")

		# Learning targets.
		self.groundTruthE_pl = tf.placeholder(shape = (self.batch_size,1), dtype = tf.float64,name="GTEs") # Energies
		self.groundTruthG_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,3), dtype = tf.float64,name="GTFs") # Forces
		self.groundTruthD_pl = tf.placeholder(shape = (self.batch_size,3), dtype = tf.float64,name="GTDs") # Dipoles.
		self.groundTruthQ_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom), dtype = tf.float64,name="GTQs") # Charges

		# Constants
		self.atom_codes = tf.Variable(self.AtomCodes,trainable=self.DoCodeLearning,dtype = self.prec,name="atom_codes")
		self.gp_tf  = tf.Variable(self.GaussParams,trainable=self.DoCodeLearning, dtype = self.prec,name="gauss_params")
		self.AvE_tf = tf.Variable(self.AverageElementEnergy, trainable=False, dtype = self.prec,name="av_energies")
		self.AvQ_tf = tf.Variable(self.AverageElementCharge, trainable=False, dtype = self.prec,name="av_charges")

		Atom1Real = tf.tile(tf.greater(self.zs_pl,0)[:,:,tf.newaxis,:],(1,1,self.MaxNeigh,1))
		nl = tf.reshape(self.nl_pl,(self.batch_size,self.MaxNAtom,self.MaxNeigh,1))
		Atom12Real = tf.logical_and(Atom1Real,tf.greater_equal(nl,0))
		Atom12Real2 = tf.tile(Atom12Real,[1,1,1,2])
		Atom12Real3 = tf.tile(Atom12Real,[1,1,1,3])
		Atom12Real4 = tf.tile(Atom12Real,[1,1,1,4])

		molis = tf.tile(tf.range(self.batch_size)[:,tf.newaxis,tf.newaxis],[1,self.MaxNAtom,self.MaxNeigh])[:,:,:,tf.newaxis]
		gather_inds0 = tf.concat([molis,nl],axis=-1)
		it1 = (self.MaxNAtom-1)*tf.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh,1),dtype=tf.int32)
		gather_inds0p = tf.concat([molis,it1],axis=-1)
		gather_inds = tf.where(Atom12Real2, gather_inds0, gather_inds0p) # Mol X MaxNatom X maxN X 2
		self.sparse_mask = tf.cast(Atom12Real,self.prec) # nmol X maxnatom X maxneigh X 1

		# sparse version of dxyzs.
		if self.DoRotGrad:
			thetas = tf.acos(2.0*tf.random_uniform([self.batch_size],dtype=tf.float64)-1.0)
			phis = tf.random_uniform([self.batch_size],dtype=tf.float64)*2*Pi
			psis = tf.random_uniform([self.batch_size],dtype=tf.float64)*2*Pi
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

		if (self.Canonicalize):
			self.cdxyzs, self.cdxyzs_p, self.CanonicalAxes =  CanonicalizeGS(self.dxyzs, zjs)
			#self.cdxyzs, self.CanonicalAxes = tf.where(Atom12Real3, CanonicalizeGS(self.dxyzs, zjs) , tf.zeros_like(diff0))
		else:
			self.cdxyzs = self.dxyzs

		if (self.DoUnChiralize):
			# force the average x,y,z coordinates to be positive.
			self.cdxyzs = RemoveReflections(self.cdxyzs)
			self.cdxyzs_p = RemoveReflections(self.cdxyzs_p)

		# Sparse Embedding.
		if 0:
			jcodes0 = tf.reshape(tf.gather(self.atom_codes,zjs),(self.batch_size,self.MaxNAtom,self.MaxNeigh,4))
			jcodes = tf.where(Atom12Real4 , jcodes0 , tf.zeros_like(jcodes0))# mol X maxNatom X maxnieh X 4
			if (not self.DoCodeLearning):
				tf.stop_gradient(jcodes)
			self.embedded = self.Embed(self.cdxyzs, jcodes, self.sparse_mask, self.gp_tf, self.l_max)
			# Sparse Energy.
			self.AtomNetEnergies,self.AtomCharges = self.AtomEmbToAtomEnergyAndCharge(self.embedded,self.zs_pl)

		self.AtomNetEnergies,self.AtomCharges = self.ChargeEmbeddedModel(self.cdxyzs, self.zs_pl, zjs,gather_inds, self.sparse_mask, self.gp_tf, self.atom_codes, self.l_max)
		if (self.Canonicalize):
			self.AtomNetEnergies_p, self.AtomCharges_p = self.ChargeEmbeddedModel(self.cdxyzs_p, self.zs_pl, zjs,gather_inds, self.sparse_mask, self.gp_tf, self.atom_codes, self.l_max)
			self.AtomNetEnergies += self.AtomNetEnergies_p
			self.AtomNetEnergies /= 2.0
			self.AtomCharges += self.AtomCharges_p
			self.AtomCharges /= 2.0

		if (self.DoChargeEmbedding or self.DoChargeLearning or self.DoDipoleLearning):
			self.MolDipoles = self.ChargeToDipole(self.xyzs_pl,self.zs_pl,self.AtomCharges)
			self.Qloss = tf.nn.l2_loss(self.AtomCharges - self.groundTruthQ_pl,name='Qloss')/tf.cast(self.batch_size*self.MaxNAtom,self.prec)
			self.Dloss = tf.nn.l2_loss(self.MolDipoles - self.groundTruthD_pl,name='Dloss')/tf.cast(self.batch_size,self.prec)
			tf.summary.scalar('Qloss',self.Qloss)
			tf.summary.scalar('Dloss',self.Dloss)
			tf.add_to_collection('losses', self.Qloss)
			tf.add_to_collection('losses', self.Dloss)
			if (self.DoChargeEmbedding):
				q2 = tf.gather_nd(self.AtomCharges,gather_inds)
				q1q2unmsk = (self.AtomCharges[:,:,tf.newaxis]*q2)
				q1q2s = tf.where(Atom12Real[:,:,:,0],q1q2unmsk,tf.zeros_like(q1q2unmsk))
				self.AtomCoulEnergies = tf.where(tf.greater(self.zs_pl,0),self.CoulombAtomEnergies(tf.norm(self.dxyzs,axis=-1),q1q2s),tf.zeros_like(self.AtomNetEnergies))
			else:
				self.AtomCoulEnergies = tf.zeros_like(self.AtomNetEnergies)
		else:
			self.AtomCoulEnergies = tf.zeros_like(self.AtomNetEnergies)

		self.MolCoulEnergies = tf.reduce_sum(self.AtomCoulEnergies,axis=1,keepdims=False)

		#self.AtomEnergies = tf.Print(self.AtomEnergies,[tf.gradients(self.AtomEnergies,self.xyzs_pl)[0]],"self.AtomEnergies",summarize=1000000)
		if (self.DoChargeEmbedding):
			self.AtomEnergies = self.AtomNetEnergies + self.AtomCoulEnergies
		else:
			self.AtomEnergies = self.AtomNetEnergies
		self.MolEnergies = tf.reduce_sum(self.AtomEnergies,axis=1,keepdims=False)

		# Optional. Verify that the canonicalized differences are invariant.
		if self.DoRotGrad:
			self.RotGrad = tf.gradients(self.MolEnergies,psis)[0]
			tf.summary.scalar('RotGrad',tf.reduce_sum(self.RotGrad))

		self.Eloss = tf.nn.l2_loss(self.MolEnergies - self.groundTruthE_pl,name='Eloss')/tf.cast(self.batch_size,self.prec)
		self.MolGradsRaw = tf.gradients(self.MolEnergies,self.xyzs_pl)[0]
		msk = tf.tile(tf.not_equal(self.zs_pl,0),[1,1,3])
		self.MolGrads = tf.where(msk,self.MolGradsRaw,tf.zeros_like(self.MolGradsRaw))
		self.MolHess = tf.hessians(self.MolEnergies,self.xyzs_pl)[0]

		t1 = tf.reshape(self.MolGrads,(self.batch_size,-1))
		t2 = tf.reshape(self.groundTruthG_pl,(self.batch_size,-1))
		diff = t1 - t2
		self.Gloss = tf.reduce_sum(tf.clip_by_value(diff*diff,1e-36,1.0))/tf.cast(self.batch_size*self.MaxNAtom*3,self.prec)
		tf.losses.add_loss(self.Gloss,loss_collection=tf.GraphKeys.LOSSES)
		tf.summary.scalar('Gloss',self.Gloss)
		tf.add_to_collection('losses', self.Gloss)

		self.Tloss = (1.0+40.0*self.Eloss)
		if (self.DoForceLearning):
			self.Tloss += (1.0+self.Gloss)
		if (self.DoDipoleLearning):
			self.Tloss += (1.0+self.Dloss)
		elif (self.DoChargeLearning):
			self.Tloss += (1.0+0.5*self.Qloss)

		tf.losses.add_loss(self.Tloss,loss_collection=tf.GraphKeys.LOSSES)
		tf.summary.scalar('ELoss',self.Eloss)
		tf.summary.scalar('TLoss',self.Tloss)
		tf.add_to_collection('losses', self.Eloss)
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

net = SparseCodedChargedGauSHNetwork(aset=None,load=True)
net.Load()
net.Train()

def MethCoords(R1,R2,R3):
	angle = 2*Pi*(35.25/360.)
	c = np.cos(angle)
	s = np.sin(angle)
	return ("""5

C  0. 0. 0.
H """+str(-R1*c)+" "+str(R1*s)+""" 0.0
H """+str(R2*c)+" "+str(R2*s)+""" 0.0
H 0.0 """+str(-R3*s)+" "+str(-R3*c)+"""
H 0.0 """+str(-R3*s)+" "+str(R3*c))


def MethCoords2(R1,R2,R3=1.):
	angle = 2*Pi*(35.25/360.)
	c = np.cos(angle)
	s = np.sin(angle)
	return ("""5

C  0. 0. 0.
H """+str(-R1+c)+" "+str(R2+s)+""" 0.0
H """+str(c)+" "+str(s)+""" 0.0
H 0.0 """+str(-R3*s)+" "+str(-R3*c)+"""
H 0.0 """+str(-R3*s)+" "+str(R3*c))

if 0:
	npts = 30
	b = MSet()
	for i in np.linspace(-.3,.3,npts):
		m = Mol()
		m.FromXYZString(MethCoords2(1.+i,1.-i,1.+i))
		b.mols.append(m)

	net = SparseCodedChargedGauSHNetwork(b)
	net.Load()
	net.Train()

	EF = net.GetEnergyForceRoutine(b.mols[-1])
	for i,d in enumerate(np.linspace(-.3,.3,npts)):
		print(d,EF(b.mols[i].coords)[0][0])
	print("------------------------")
	for i,d in enumerate(np.linspace(-.3,.3,npts)):
		print(d,EF(b.mols[i].coords)[1])

if 0:
	mi = np.random.randint(len(b.mols))
	m = b.mols[mi]
	print(m.atoms, m.coords)
	EF = net.GetEnergyForceRoutine(m)
	print(EF(m.coords))
	Opt = GeomOptimizer(EF)
	m=Opt.OptGD(m)
	m.Distort(0.2)
	m=Opt.OptGD(m,"FromDistorted")

if 1:
	from matplotlib import pyplot as plt
	import matplotlib.cm as cm
	m = Mol()
	m.FromXYZString(MethCoords(1.,1.,1.))
	EF = net.GetEnergyForceRoutine(m)

	Opt = GeomOptimizer(EF)
	m=Opt.OptGD(m,"YYY")

	atomnumber = 0
	nx, ny = 80, 80
	x = np.linspace(-.2, .2, nx)
	y = np.linspace(-.2, .2, ny)
	X, Y = np.meshgrid(x, y)
	fig = plt.figure()
	ax = fig.add_subplot(111)

	Ens = np.zeros_like(X)
	Fx = np.zeros_like(X)
	Fy = np.zeros_like(X)

	for xi in range(X.shape[0]):
		for yi in range(X.shape[1]):
			ctmp = m.coords.copy()
			ctmp[atomnumber][0]+=X[xi,yi]
			ctmp[atomnumber][1]+=Y[xi,yi]
			e,f = EF(ctmp)
			f/=-JOULEPERHARTREE
			Ens[xi,yi] = e
			Fx[xi,yi] = f[atomnumber,0]
			Fy[xi,yi] = f[atomnumber,1]

	color = 2 * np.log(np.hypot(Fx, Fy))
	ax.streamplot(x, y, Fx, Fy, color=color, linewidth=1, cmap=plt.cm.inferno,
		density=2, arrowstyle='->', arrowsize=1.5)
	plt.pcolormesh(X, Y, Ens, cmap = cm.gray)
	plt.show()

# Some code to find and visualize largest errors in the set.
