"""
This version adds Coulombic Long-Range Forces.
The old file is intended to be deleted after everything works here.
"""

from TensorMol import *
import numpy as np

b = MSet("HNCO_small")
#b = MSet("chemspider12_clean_maxatom35")
#b = MSet("chemspider20_1_opt_all")
b.Load()
# This removes any linear-stoichiometric contribution from these
# properties to speed learning by keeping these as small as possible.
# This network will specifically target these renomalized properties appropriately.
#c.Load()
#b.mols=b.mols+(c.mols[:int(len(b.mols)*0.5)])
MAX_ATOMIC_NUMBER = 55

def sftpluswparam(x):
	return tf.log(1.0+tf.exp(100.*x))/100.0

def safe_inv_norm(x_):
	nrm = tf.clip_by_value(tf.norm(x_,axis=-1,keepdims=True),1e-36,1e36)
	nrm_ok = tf.not_equal(nrm,0.)
	safe_nrm = tf.where(nrm_ok,nrm,tf.ones_like(nrm))
	return tf.where(nrm_ok,1.0/safe_nrm,tf.zeros_like(nrm))

def polykern(r):
	"""
	Polynomial cutoff 1/r (in BOHR) obeying:
	kern = 1/r at SROuter and LRInner
	d(kern) = d(1/r) (true force) at SROuter,LRInner
	d**2(kern) = d**2(1/r) at SROuter and LRInner.
	d(kern) = 0 (no force) at/beyond SRInner and LROuter

	The hard cutoff is 15 Angstrom.
	"""
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

def CanonicalizeGS(dxyzs):
	"""
	Canonicalize using nearest three atoms and Graham-Schmidt.
	If there are not three linearly independent atoms within
	4A, the output will not be rotationally invariant, although
	The axes will still be as invariant as possible.

	The axes are also smooth WRT radial displacments, because they are
	smoothly mixed with each other.

	Args:
		dxyz: a nMol X maxNatom X maxNatom X 3 tensor of atoms. (differenced from center of embedding
		ie: ... X i X i = (0.,0.,0.))
	"""
	# Append orthogonal axes to dxyzs
	argshape = tf.shape(dxyzs)
	defaultAxes = tf.tile(tf.reshape(4.0*tf.eye(3,dtype=tf.float64),(1,1,3,3)),[argshape[0],argshape[1],1,1])
	dxyzsandDef = tf.concat([dxyzs,defaultAxes],axis=2)

	realdata = tf.reshape(dxyzs,(argshape[0]*argshape[1],argshape[2],3))
	togather = tf.reshape(dxyzsandDef,(argshape[0]*argshape[1],argshape[2]+3,3))

	nrm = tf.clip_by_value(-1.0*tf.norm(dxyzsandDef,axis=-1),1e-36,1e36)
	nrm_ok = tf.not_equal(nrm,0.)
	safe_nrm = tf.where(nrm_ok,nrm,tf.ones_like(nrm))
	weights = tf.where(nrm_ok,tf.exp(nrm),tf.zeros_like(nrm)) # Mol X MaxNAtom X MaxNAtom
	maskedDs = tf.where(tf.equal(weights,1.),tf.zeros_like(weights),weights)
	# GS orth the first three vectors.
	vals, inds = tf.nn.top_k(maskedDs,k=3)
	inds = tf.reshape(inds,(argshape[0]*argshape[1],3))
	vals = tf.reshape(vals,(argshape[0]*argshape[1],3))
	v1i = tf.concat([tf.range(argshape[0]*argshape[1])[:,tf.newaxis],inds[:,:1]],axis=-1)
	v2i = tf.concat([tf.range(argshape[0]*argshape[1])[:,tf.newaxis],inds[:,1:2]],axis=-1)
	v3i = tf.concat([tf.range(argshape[0]*argshape[1])[:,tf.newaxis],inds[:,2:3]],axis=-1)
	v10 = tf.gather_nd(togather,v1i)
	v20 = tf.gather_nd(togather,v2i)
	v30 = tf.gather_nd(togather,v3i)
	w1 = tf.exp(-tf.clip_by_value(tf.norm(v10,axis=-1,keepdims=True),1e-36,1e36))
	w2 = tf.exp(-tf.clip_by_value(tf.norm(v20,axis=-1,keepdims=True),1e-36,1e36))
	w3 = tf.exp(-tf.clip_by_value(tf.norm(v30,axis=-1,keepdims=True),1e-36,1e36))
	v1 = w1*v10 + w2*v20
	v1 *= safe_inv_norm(v1)
	v2 = w2*v10 + w1*v20 + w3*v30
	v2 -= tf.einsum('ij,ij->i',v1,v2)[:,tf.newaxis]*v1
	v2 *= safe_inv_norm(v2)
	v3 = w2*v20 + w3*v30
	v3 -= tf.einsum('ij,ij->i',v1,v3)[:,tf.newaxis]*v1
	v3 -= tf.einsum('ij,ij->i',v2,v3)[:,tf.newaxis]*v2
	v3 *= safe_inv_norm(v3)
	vs = tf.concat([v1[:,tf.newaxis,:],v2[:,tf.newaxis,:],v3[:,tf.newaxis,:]],axis=1)
	tore = tf.einsum('ijk,ilk->ijl',realdata,vs)
	return tf.reshape(tore,tf.shape(dxyzs))

class SparseCodedChargedGauSHNetwork:
	"""
	This is the basic TensorMol0.2 model chemistry.
	"""
	def __init__(self,aset=None):
		self.prec = tf.float64
		self.batch_size = 128 # Force learning strongly modulates what you can do.
		self.MaxNAtom = 32
		self.MaxNeigh = self.MaxNAtom
		self.learning_rate = 0.00005
		self.AtomCodes = ELEMENTCODES #np.random.random(size=(MAX_ATOMIC_NUMBER,4))
		self.AtomTypes = [1,6,7,8]
		self.l_max = 3
		#self.GaussParams = np.array([[0.35, 0.30], [0.70, 0.30], [1.05, 0.30], [1.40, 0.30], [1.75, 0.30], [2.10, 0.30], [2.45, 0.30],[2.80, 0.30], [3.15, 0.30], [3.50, 0.30], [3.85, 0.30], [4.20, 0.30], [4.55, 0.30], [4.90, 0.30]])
		#self.GaussParams = np.array([[0.36, 0.25], [0.70, 0.24], [1.05, 0.24], [1.38, 0.23], [1.70, 0.23],[2.08, 0.23], [2.79, 0.23], [2.42, 0.23],[3.14, 0.23], [3.50, 0.23], [3.85, 0.23], [4.20, 0.23], [4.90, 0.23], [5.50, 0.22], [6.0, 0.22]])
		self.GaussParams = np.array([[ 0.37477018,  0.25175677],[ 0.67658861,  0.23472445], [ 1.05008962,  0.23588795],[ 1.38640627,  0.22124612], [ 1.68125033,  0.21762672], [ 2.05397151,  0.21847124], [ 2.79472851,  0.15731322], [ 3.13242662,  0.19378809], [ 3.80189948,  0.18397461], [ 4.89845145,  0.13036654], [ 5.50038598,  0.11493009]])
		self.nrad = len(self.GaussParams)
		self.nang = (self.l_max+1)**2
		self.ncodes = self.AtomCodes.shape[-1]
		self.ngaush = self.nrad*self.nang
		self.nembdim = self.ngaush*self.ncodes
		self.mset = aset
		if (aset != None):
			self.MaxNAtom = b.MaxNAtom()
			self.AtomTypes = b.AtomTypes()
			AvE,AvQ = aset.RemoveElementAverages()
			self.AverageElementEnergy = np.zeros((MAX_ATOMIC_NUMBER))
			self.AverageElementCharge = np.zeros((MAX_ATOMIC_NUMBER))
			for ele in AvE.keys():
				self.AverageElementEnergy[ele] = AvE[ele]
				self.AverageElementCharge[ele] = AvQ[ele]
			# Get a reasonable number of neighbors.
			xyzs = np.zeros((100,self.MaxNAtom,3))
			zs = np.zeros((100,self.MaxNAtom,1))
			for i in range(100):
				mi = np.random.randint(len(aset.mols))
				m = aset.mols[mi]
				xyzs[i,:m.NAtoms()] = m.coords
				zs[i,:m.NAtoms(),0] = m.atoms
			NLT = self.NLTensors(xyzs,zs)
			self.MaxNeigh = NLT.shape[2]+2
			print("self.MaxNeigh = ", self.MaxNeigh)
		self.sess = None
		self.Prepare()
		return

	def Load(self):
		chkpt = tf.train.latest_checkpoint('./networks/')
		self.saver.restore(self.sess, chkpt)
		return

	def GetEnergyForceRoutine(self,m):
		if (m.NAtoms() > self.MaxNAtom):
			self.MaxNAtom = m.NAtoms()
		self.batch_size=1
		self.Prepare()
		self.Load()
		def EF(xyz_,DoForce=True):
			xyzs = np.zeros((self.batch_size,self.MaxNAtom,3))
			Zs = np.zeros((self.batch_size,self.MaxNAtom,1))
			nls = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh),dtype=np.int32)
			xyzs[0,:m.NAtoms(),:] = xyz_
			Zs[0,:m.NAtoms(),0] = m.atoms
			nlt = self.NLTensors(xyzs,Zs)
			nls[:nlt.shape[0],:nlt.shape[1],:nlt.shape[2]] = nlt
			feed_dict = {self.xyzs_pl:xyzs, self.zs_pl:Zs,self.nl_pl:nls}
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
			Zs = np.zeros((self.batch_size,self.MaxNAtom,1))
			nls = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh),dtype=np.int32)
			xyzs[0,:m.NAtoms(),:] = xyz_
			Zs[0,:m.NAtoms(),0] = m.atoms
			nlt = self.NLTensors(xyzs,Zs)
			nls[:nlt.shape[0],:nlt.shape[1],:nlt.shape[2]] = nlt
			feed_dict = {self.xyzs_pl:xyzs, self.zs_pl:Zs,self.nl_pl:nls}
			ens,fs,hs = self.sess.run([self.MolEnergies,self.MolGrads,self.MolHess], feed_dict=feed_dict)
			return ens[0], fs[0][:m.NAtoms()]*(-JOULEPERHARTREE), hs[0][:m.NAtoms()][:m.NAtoms()]*JOULEPERHARTREE*JOULEPERHARTREE
		return EFH

	def NextBatch(self,aset):
		# Randomly accumulate a batch.
		xyzs = np.zeros((self.batch_size,self.MaxNAtom,3),dtype=np.float)
		true_force = np.zeros((self.batch_size,self.MaxNAtom,3),dtype=np.float)
		zs = np.zeros((self.batch_size,self.MaxNAtom),dtype=np.int32)
		nls = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh),dtype=np.int32)
		true_ae = np.zeros((self.batch_size,1),dtype=np.float)
		qs = np.zeros((self.batch_size,self.MaxNAtom),dtype=np.float64) # Charges.
		ds = np.zeros((self.batch_size,3),dtype=np.float64) # Dipoles.
		for i in range(self.batch_size):
			mi = np.random.randint(len(aset.mols))
			m = aset.mols[mi]
			xyzs[i,:m.NAtoms()] = m.coords
			zs[i,:m.NAtoms()] = m.atoms
			true_ae[i]=m.properties["energy"]
			true_force[i,:m.NAtoms()]=m.properties["gradients"]
			qs[i,:m.NAtoms()]=m.properties["charges"]
			ds[i]=m.properties["dipole"]
		nlt = self.NLTensors(xyzs,zs)
		nls[:nlt.shape[0],:nlt.shape[1],:nlt.shape[2]] = nlt
		return {self.xyzs_pl:xyzs, self.zs_pl:zs[:,:,np.newaxis],self.nl_pl:nls,self.groundTruthE_pl:true_ae, self.groundTruthG_pl:true_force, self.groundTruthQ_pl:qs, self.groundTruthD_pl:ds}

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

	def NLTensors(self, xyzs_, zs_, Rcut = 15.0, ntodo_=None):
		"""
		Generate Neighborlist arrays for sparse version.

		Args:
			xyzs_ : a coordinate tensor nmol X Maxnatom X 3
			Zs_ : a AN tensor nmol X Maxnatom X 1
			ntodo_: For periodic (unused.)
		Returns:
			nlarray a nmol X Maxnatom X MaxNeighbors int32 array.
				which is -1 = blank, 0 = atom zero is a neighbor within Rcut
		"""
		RawLists=[]
		for mi in range(xyzs_.shape[0]):
			RawLists.append(Make_NListNaive(xyzs_[mi],Rcut,self.MaxNAtom,True))
		maxneigh = 0
		for rl in RawLists:
			for a in rl:
				if len(a) > maxneigh:
					maxneigh = len(a)
		tore = np.ones((xyzs_.shape[0],xyzs_.shape[1],maxneigh),dtype=np.int32)
		tore *= -1
		for i,rl in enumerate(RawLists):
			for j,a in enumerate(rl):
				if (zs_[i,j] != 0):
					for k,l in enumerate(a):
						tore[i,j,k] = l
		#print tore
		return tore

	def CoulombEnergies(self,dxyzs_,q1q2s_):
		"""
		Atom Coulomb energy calculated

		Args:
			dxyzs_: (nmol X MaxNAtom X MaxNeigh X 1) distances tensor (angstrom)
			q1q2s_: charges (nmol X MaxNAtom X MaxNeigh X 1) charges tensor. (atomic)
		Returns:
			 (nmol X 1) tensor of energies (atomic)
		"""
		return (1.0/2.0)*tf.reduce_sum(polykern(dxyzs_*1.889725989)*q1q2s_,axis=(1,2))[:,tf.newaxis]

	def ChargeToDipole(self,xyzs_,zs_,qs_):
		"""
		Calculate the dipole moment relative to center of atom.
		"""
		n_atoms = tf.clip_by_value(tf.cast(tf.reduce_sum(zs_,axis=(1,2)),tf.float64),1e-36,1e36)
		COA = tf.reduce_sum(xyzs_,axis=1)/n_atoms[:,tf.newaxis]
		return tf.reduce_sum((xyzs_ - COA[:,tf.newaxis,:])*qs_[:,:,tf.newaxis],axis=1)

	def AtomEmbToAtomEnergy(self,emb,Zs):
		"""
		This version creates a network to integrate weight information.
		and then works like any ordinary network.
		NOTE: This network is universal in the sense that it works on ANY atom!

		Args:
			emb: # mol X maxNAtom X ang X rad X code tensor.
			Zs: mol X maxNatom X 1 atomic number tensor.

		Returns:
			mol X maxNatom X 1 tensor of atom energies.
			these include constant shifts.
		"""
		ncase = self.batch_size*self.MaxNAtom
		Zrs = tf.cast(tf.reshape(Zs,(ncase,-1)),self.prec)
		nchan = self.AtomCodes.shape[1]
		nembdim = self.nembdim
		AvEs = tf.reshape(tf.gather(self.AvE_tf, Zs, axis=0),(self.batch_size,self.MaxNAtom,1)) # (mol * maxNatom) X 1
		CODES = tf.reshape(tf.gather(self.atom_codes, Zs, axis=0),(ncase,nchan)) # (mol * maxNatom) X 4
		# Combine the codes of the main atom and the sensed atom
		# Using a hinton-esque tensor decomposition.
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
		l1 = tf.layers.dense(inputs=l0p,units=512,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer)
		l1p = tf.concat([l1,CODES],axis=-1)
		l2 = tf.layers.dense(inputs=l1p,units=512,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer)
		# in the final layer use the atom code information.
		l2p = tf.concat([l2,CODES],axis=-1)
		l3 = tf.layers.dense(l2p,units=1,activation=None,use_bias=True)*msk
		# Finally allow for a simple 1-D linear filter based on element type.
		return tf.reshape(l3,(self.batch_size,self.MaxNAtom,1))+AvEs

	def AtomEmbToAtomCharge(self,emb,Zs):
		"""
		Zero out the total charge.

		Args:
			emb: # mol X maxNAtom X ang X rad X code tensor.
			Zs: mol X maxNatom X 1 atomic number tensor.

		Returns:
			(atom energies, atom charges)
		"""
		ncase = self.batch_size*self.MaxNAtom
		Zrs = tf.cast(tf.reshape(Zs,(ncase,-1)),self.prec)
		nchan = self.AtomCodes.shape[1]
		nembdim = self.nembdim
		AvQs = tf.reshape(tf.gather(self.AvQ_tf, Zs, axis=0),(self.batch_size,self.MaxNAtom)) # (mol * maxNatom) X 1
		CODES = tf.reshape(tf.gather(self.atom_codes, Zs, axis=0),(ncase,nchan)) # (mol * maxNatom) X 4
		# Combine the codes of the main atom and the sensed atom
		# Using a hinton-esque tensor decomposition.
		CODEKERN1 = tf.get_variable(name="QCodeKernel", shape=(nchan,nchan),dtype=self.prec)
		CODEKERN2 = tf.get_variable(name="QCodeKernel2", shape=(nchan,nchan),dtype=self.prec)
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
		l1 = tf.layers.dense(inputs=l0p,units=512,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer)
		l1p = tf.concat([l1,CODES],axis=-1)
		l2 = tf.layers.dense(inputs=l1p,units=512,activation=sftpluswparam,use_bias=True, kernel_initializer=tf.variance_scaling_initializer, bias_initializer=tf.variance_scaling_initializer)
		# in the final layer use the atom code information.
		l2p = tf.concat([l2,CODES],axis=-1)
		l3 = tf.layers.dense(l2p,units=1,activation=None,use_bias=True)*msk
		charges = tf.reshape(l3,(self.batch_size,self.MaxNAtom))
		# Set the total charges to neutral by evenly distributing any excess charge.
		excess_charges = tf.reduce_sum(charges,axis=[1])
		n_atoms = tf.reduce_sum(tf.where(tf.equal(Zs,0),Zs,tf.ones_like(Zs)),axis=[1,2])
		fix = -1.0*excess_charges/tf.cast(n_atoms,tf.float64)
		return charges + fix[:,tf.newaxis] + AvQs

	def train_step(self,step):
		feed_dict = self.NextBatch(self.mset)
		_,train_loss = self.sess.run([self.train_op, self.Tloss], feed_dict=feed_dict)
		self.print_training(step, train_loss)
		return

	def print_training(self, step, loss_):
		if (step%10==0):
			self.saver.save(self.sess, './networks/SparseCodedGauSH',global_step=step)
			print("step: ", "%7d"%step, "  train loss: ", "%.10f"%(float(loss_)))
			#print("Gauss Params: ",self.sess.run([self.gp_tf])[0])
			#print("AtomCodes: ",self.sess.run([self.atom_codes])[0])
			feed_dict = self.NextBatch(self.mset)
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
			print("Mean Abs Error: (Energy)", np.average(np.abs(ens-feed_dict[self.groundTruthE_pl])))
			print("Mean Abs Error (Force): ", np.average(np.abs(frcs-feed_dict[self.groundTruthG_pl])))
			if (self.DoDipoleLearning):
				print("Mean Abs Error (Dipole): ", np.average(np.abs(dipoles-feed_dict[self.groundTruthD_pl])))
			if (self.DoChargeLearning):
				print("Mean Abs Error (Charges): ", np.average(np.abs(charges-feed_dict[self.groundTruthQ_pl])))
			if (self.DoRotGrad):
				print("RotGrad:",self.sess.run([self.RotGrad], feed_dict=feed_dict))
			self.writer.add_summary(summary,step)
		return

	def training(self, loss):
		#optimizer = tf.train.AdamOptimizer(self.learning_rate)
		#train_op = optimizer.minimize(loss, global_step=global_step)
		optimizer = tf.train.AdamOptimizer(learning_rate=(self.learning_rate))
		gvs = optimizer.compute_gradients(loss)
		capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
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
		self.DoCodeLearning = False
		self.DoDipoleLearning = False
		self.DoChargeLearning = True
		self.DoChargeEmbedding = True

		self.xyzs_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,3), dtype = self.prec)
		self.zs_pl = tf.placeholder(shape = (self.batch_size, self.MaxNAtom, 1), dtype = tf.int32)
		self.nl_pl = tf.placeholder(shape = (self.batch_size, self.MaxNAtom, self.MaxNeigh), dtype = tf.int32)
		tf.stop_gradient(self.nl_pl)
		tf.stop_gradient(self.zs_pl)

		# Learning targets.
		self.groundTruthE_pl = tf.placeholder(shape = (self.batch_size,1), dtype = tf.float64) # Energies
		self.groundTruthG_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,3), dtype = tf.float64) # Forces
		self.groundTruthD_pl = tf.placeholder(shape = (self.batch_size,3), dtype = tf.float64) # Dipoles.
		self.groundTruthQ_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom), dtype = tf.float64) # Charges

		# Constants
		self.atom_codes = tf.Variable(self.AtomCodes,trainable=self.DoCodeLearning)
		self.gp_tf  = tf.Variable(self.GaussParams,trainable=self.DoCodeLearning, dtype = self.prec)
		self.AvE_tf = tf.Variable(self.AverageElementEnergy, trainable=False, dtype = self.prec)
		self.AvQ_tf = tf.Variable(self.AverageElementCharge, trainable=False, dtype = self.prec)

		# self.dxyzs now has shape nmol X maxnatom X maxneigh
		maskatom1 = tf.reshape(tf.where(tf.not_equal(self.zs_pl,0),tf.ones_like(self.zs_pl),self.zs_pl),(self.batch_size,self.MaxNAtom,1,1))
		nl = tf.reshape(self.nl_pl,(self.batch_size,self.MaxNAtom,self.MaxNeigh,1))
		#nl = tf.Print(nl,[nl],"nl",summarize=1000000)
		molis = tf.tile(tf.range(self.batch_size)[:,tf.newaxis,tf.newaxis],[1,self.MaxNAtom,self.MaxNeigh])[:,:,:,tf.newaxis]
		# Keep the negative one encoding.
		molis = tf.where(tf.equal(nl,-1),-1*tf.ones_like(molis),molis)
		gathis = tf.concat([molis,nl],axis=-1) # Mol X MaxNatom X maxN X 2
		#gathis = tf.Print(gathis,[gathis],"gathis",summarize=1000000)
		sparse_maski = tf.where(tf.equal(gathis[:,:,:,1:],-1),tf.zeros_like(nl),tf.ones_like(nl))*maskatom1
		gathis *= sparse_maski
		self.sparse_mask = tf.cast(sparse_maski,self.prec) # nmol X maxnatom X maxneigh X 1
		boolmsk1 = tf.cast(sparse_maski,tf.bool)
		boolmsk3 = tf.tile(boolmsk1,[1,1,1,3])
		#self.sparse_mask = tf.Print(self.sparse_mask,[self.sparse_mask],"Sparse Mask",summarize=1000000)
		tf.stop_gradient(self.sparse_mask)
		tf.stop_gradient(sparse_maski)
		tf.stop_gradient(gathis)

		# sparse version of dxyzs.
		nxs = tf.gather_nd(self.xyzs_pl,gathis) # mol X maxNatom X maxneigh X 3
		zxs0 = tf.gather_nd(self.zs_pl,gathis)
		zxs = tf.where(boolmsk1,zxs0,tf.zeros_like(zxs0)) # mol X maxNatom X maxneigh X 1
		coord1 = tf.expand_dims(self.xyzs_pl, axis=2) # mol X maxnatom X 1 X 3
		coord2 = tf.gather_nd(self.xyzs_pl,gathis)
		diff0 = (coord1-coord2)
		self.dxyzs = tf.where(boolmsk3, diff0, tf.zeros_like(diff0))
		# Canonicalized difference Vectors.
		self.cdxyzs = tf.where(boolmsk3, CanonicalizeGS(self.dxyzs) , tf.zeros_like(diff0))

		# Sparse Embedding.
		jcodes = tf.reshape(tf.gather(self.atom_codes,zxs),(self.batch_size,self.MaxNAtom,self.MaxNeigh,4))*self.sparse_mask # mol X maxNatom X maxnieh X 4
		if (not self.DoCodeLearning):
			tf.stop_gradient(jcodes)
		self.embedded = self.Embed(self.cdxyzs, jcodes, self.sparse_mask, self.gp_tf, self.l_max)
		# Sparse Energy.
		self.AtomEnergies = self.AtomEmbToAtomEnergy(self.embedded,self.zs_pl)
		self.MolEnergies = tf.reduce_sum(self.AtomEnergies,axis=1,keepdims=False)

		if (self.DoChargeEmbedding or self.DoChargeLearning or self.DoDipoleLearning):
			self.AtomCharges = self.AtomEmbToAtomCharge(self.embedded,self.zs_pl) # (nmol X maxnatom X 1)
			self.MolDipoles = self.ChargeToDipole(self.xyzs_pl,self.zs_pl,self.AtomCharges)
			self.Qloss = tf.nn.l2_loss(self.AtomCharges - self.groundTruthQ_pl,name='Qloss')/tf.cast(self.batch_size,self.prec)
			self.Dloss = tf.nn.l2_loss(self.MolDipoles - self.groundTruthD_pl,name='Dloss')/tf.cast(self.batch_size,self.prec)
			tf.summary.scalar('Qloss',self.Qloss)
			tf.summary.scalar('Dloss',self.Dloss)
			tf.add_to_collection('losses', self.Qloss)
			tf.add_to_collection('losses', self.Dloss)
			if (self.DoChargeEmbedding):
				q2 = tf.gather_nd(self.AtomCharges,gathis)
				q1q2unmsk = (self.AtomCharges[:,:,tf.newaxis]*q2)
				q1q2s = tf.where(boolmsk1[:,:,:,0],q1q2unmsk,tf.zeros_like(q1q2unmsk))
				self.MolCoulEnergies = self.CoulombEnergies(tf.norm(self.dxyzs,axis=-1),q1q2s)
			else:
				self.MolCoulEnergies = tf.zeros_like(self.MolEnergies)
		else:
			self.MolCoulEnergies = tf.zeros_like(self.MolEnergies)

		#self.AtomEnergies = tf.Print(self.AtomEnergies,[tf.gradients(self.AtomEnergies,self.xyzs_pl)[0]],"self.AtomEnergies",summarize=1000000)
		if (self.DoChargeEmbedding):
			self.MolEnergies += self.MolCoulEnergies

		# Optional. Verify that the canonicalized differences are invariant.
		if self.DoRotGrad:
			self.RotGrad = tf.gradients(self.embedded,psis)[0]
			tf.summary.scalar('RotGrad',tf.reduce_sum(self.RotGrad))

		# Add force error?
		self.Eloss = tf.nn.l2_loss(self.MolEnergies - self.groundTruthE_pl,name='Eloss')/tf.cast(self.batch_size,self.prec)
		self.MolGrads = tf.gradients(self.MolEnergies,self.xyzs_pl)[0]
		self.MolHess = tf.hessians(self.MolEnergies,self.xyzs_pl)[0]

		if (self.DoForceLearning):
			t1 = tf.reshape(self.MolGrads,(self.batch_size,-1))
			t2 = tf.reshape(self.groundTruthG_pl,(self.batch_size,-1))
			nrm1 = tf.sqrt(tf.clip_by_value(tf.reduce_sum(t1*t1,axis=1),1e-36,1e36))
			nrm2 = tf.sqrt(tf.clip_by_value(tf.reduce_sum(t2*t2,axis=1),1e-36,1e36))
			diff = nrm1-nrm2
			num = tf.reduce_sum(t1*t2,axis=1)
			self.Gloss1 = (1.0 - tf.reduce_mean(num/(nrm1*nrm2)))/20.
			self.Gloss2 = (tf.reduce_mean(diff*diff))/100.
			#self.Gloss = tf.losses.mean_squared_error(self.MolGrads, self.groundTruthG_pl)
			tf.summary.scalar('GLossDir',self.Gloss1)
			tf.summary.scalar('GLossMag',self.Gloss2)
			tf.add_to_collection('losses', self.Gloss1)
			tf.add_to_collection('losses', self.Gloss2)
			self.Tloss = self.Eloss + self.Gloss1 + self.Gloss2
		else:
			self.Tloss = self.Eloss

		if (self.DoDipoleLearning):
			self.Tloss += self.Dloss
		elif (self.DoChargeLearning):
			self.Tloss += self.Qloss/150.

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

		if (True):
			print("logging with FULL TRACE")
			self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			self.run_metadata = tf.RunMetadata()
			self.writer.add_run_metadata(self.run_metadata, "init", global_step=None)
		else:
			self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			self.run_metadata = tf.RunMetadata()
			self.writer.add_run_metadata(self.run_metadata, "init", global_step=None)

		self.sess.run(self.init)
		#self.sess.graph.finalize()

net = SparseCodedChargedGauSHNetwork(b)
#net.Load()
net.Train()
if 0:
	mi = np.random.randint(len(b.mols))
	m = b.mols[mi]
	print(m.atoms, m.coords)
	EF = net.GetEnergyForceRoutine(m)
	print(EF(m.coords))
	Opt = GeomOptimizer(EF)
	Opt.Opt(m)
