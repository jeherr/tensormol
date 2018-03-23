from TensorMol import *
import numpy as np

b = MSet("HNCO_small")
b.Load()
MAX_ATOMIC_NUMBER = 55

def MaskedReduceMean(masked,pair_mask,axis=-2):
	num = tf.reduce_sum(masked,axis=-2,keepdims=True)
	denom = tf.reduce_sum(pair_mask,axis=-2,keepdims=True)+1e-15
	return num/denom

def sftpluswparam(x):
	return tf.log(1.0+tf.exp(100.*x))/100.0

def BuildMultiplicativeZeroMask(Tensor):
 	return tf.where(tf.greater(Tensor,0.0),tf.ones_like(Tensor),tf.zeros_like(Tensor))

def CanonicalizePCA(dxyzs, pair_mask ,ChiralInv=True):
	"""
	Perform a PCA to create invariant axes.
	These axes are invariant to both rotation and reflection.
	MaxNAtom must be >= 4 otherwise this won't work.
	I have tested the rotational invariance and differentiability of this routine

	Args:
	    dxyz: a nMol X maxNatom X maxNatom X 3 tensor of atoms. (differenced from center of embedding
				ie: ... X i X i = (0.,0.,0.))
	Returns:
	    Cdxyz: canonically oriented versions of the above coordinates.
	"""
	#cutoffd = tf.reciprocal(dxyzs+1.0) #tf.exp(-1.0*dxyzs)
	masked = dxyzs*pair_mask
	ap = (masked - MaskedReduceMean(masked,pair_mask))*pair_mask
	C = tf.einsum('lmji,lmjk->lmik',ap,ap) # Covariance matrix.
	w,v = tf.self_adjoint_eig(C)
	tore = tf.matmul(dxyzs,v)*pair_mask
	if (not ChiralInv):
		return tore
	signc = tf.sign(tf.reduce_mean(tore,axis=-2,keepdims=True))
	# output axes only match up to a sign due to phase freedom of eigenvalues.
	# Make a convention that mean axis is positive.
	tore2 = tore*signc
	return tore2

def CanonicalizeGS(dxyzs):
	"""
	Canonicalize using nearest three atoms and Graham-Schmidt.

	Args:
		dxyz: a nMol X maxNatom X maxNatom X 3 tensor of atoms. (differenced from center of embedding
		ie: ... X i X i = (0.,0.,0.))
	"""
	argshape = tf.shape(dxyzs)
	togather = tf.reshape(dxyzs,(argshape[0]*argshape[1],argshape[1],3))
	weights = tf.exp(-1.0*tf.norm(dxyzs,axis=-1))
	maskedDs = tf.where(tf.equal(weights,1.),tf.zeros_like(weights),weights)
	# GS orth the first three vectors.
	tosort= tf.reshape(maskedDs,(argshape[0]*argshape[1],-1))
	vals, inds = tf.nn.top_k(maskedDs,k=4)
	inds = tf.reshape(inds,(argshape[0]*argshape[1],4))
	v1i = tf.concat([tf.range(argshape[0]*argshape[1])[:,tf.newaxis],inds[:,:1]],axis=-1)
	v2i = tf.concat([tf.range(argshape[0]*argshape[1])[:,tf.newaxis],inds[:,1:2]],axis=-1)
	v3i = tf.concat([tf.range(argshape[0]*argshape[1])[:,tf.newaxis],inds[:,2:3]],axis=-1)
	v1 = tf.gather_nd(togather,v1i)
	v1 /= tf.norm(v1,axis=-1,keepdims=True)+1e-36
	v2 = tf.gather_nd(togather,v2i)
	v2 -= tf.einsum('ij,ij->i',v1,v2)[:,tf.newaxis]*v1
	v2 /= tf.norm(v2,axis=-1,keepdims=True)+1e-36
	v3 = tf.gather_nd(togather,v3i)
	v3 -= tf.einsum('ij,ij->i',v1,v3)[:,tf.newaxis]*v1
	v3 -= tf.einsum('ij,ij->i',v2,v3)[:,tf.newaxis]*v2
	v3 /= tf.norm(v3,axis=-1,keepdims=True)+1e-36
	vs = tf.concat([v1[:,tf.newaxis,:],v2[:,tf.newaxis,:],v3[:,tf.newaxis,:]],axis=1)
	tore = tf.einsum('ijk,ilk->ijl',togather,vs)
	return tf.reshape(tore,tf.shape(dxyzs))

class InGauShBPNetwork:
	def __init__(self,aset=None):
		self.prec = tf.float64
		self.batch_size = 256
		self.MaxNAtom = 32
		self.learning_rate = 0.0005
		self.AtomCodes = ELEMENTCODES #np.random.random(size=(MAX_ATOMIC_NUMBER,4))
		self.AtomTypes = [1,6,7,8]
		self.l_max = 3
		#self.GaussParams = np.array([[0.35, 0.30], [0.70, 0.30], [1.05, 0.30], [1.40, 0.30], [1.75, 0.30], [2.10, 0.30], [2.45, 0.30],[2.80, 0.30], [3.15, 0.30], [3.50, 0.30], [3.85, 0.30], [4.20, 0.30], [4.55, 0.30], [4.90, 0.30]])
		#self.GaussParams = np.array([[0.35, 0.35], [0.70, 0.35], [1.05, 0.35], [1.40, 0.35], [2.10, 0.35], [2.80, 0.35], [3.15, 0.35], [3.50, 0.35], [3.85, 0.35], [4.20, 0.35], [4.90, 0.35]])
		self.GaussParams = np.array([[0.70, 0.30], [1.05, 0.30], [1.40, 0.30], [2.10, 0.30],[2.80, 0.30],[3.50, 0.30], [4.20, 0.30], [4.90, 0.30], [5.50, 0.30]])
		self.mset = aset
		if (aset != None):
			self.MaxNAtom = b.MaxNAtom()
			self.AtomTypes = b.AtomTypes()
		self.sess = None
		self.Prepare()
		return

	def NextBatch(self,aset):
		# Randomly accumulate a batch.
		xyzs = np.zeros((self.batch_size,self.MaxNAtom,3),dtype=np.float)
		true_force = np.zeros((self.batch_size,self.MaxNAtom,3),dtype=np.float)
		zs = np.zeros((self.batch_size,self.MaxNAtom),dtype=np.int32)
		true_ae = np.zeros((self.batch_size,1),dtype=np.float)
		for i in range(self.batch_size):
			mi = np.random.randint(len(b.mols))
			m = b.mols[mi]
			xyzs[i,:m.NAtoms()] = m.coords
			zs[i,:m.NAtoms()] = m.atoms
			true_ae[i]=m.properties["atomization"]
			true_force[i,:m.NAtoms()]=m.properties["gradients"]
		return {self.xyzs_pl:xyzs, self.zs_pl:zs[:,:,np.newaxis],self.groundTruthE_pl:true_ae, self.groundTruthG_pl:true_force}

	def Embed(self, dxyzs, Zs, pair_mask, gauss_params, elecode, l_max):
		"""
		Returns the GauSH embedding of every atom.
		"""
		dist_tensor = tf.norm(dxyzs+1.e-36,axis=-1)
		# NMOL X MAXNATOM X MAXNATOM X NSH
		SH = tf_spherical_harmonics(dxyzs, dist_tensor, l_max)*pair_mask
		RAD = tf_gauss(dist_tensor, gauss_params)*pair_mask
		# Perform each of the contractions.
		SHRAD = tf.einsum('mijk,mijl->mijkl',SH,RAD)
		CODES = tf.reshape(tf.gather(elecode, Zs, axis=0),(self.batch_size,self.MaxNAtom,-1)) # mol X maxNatom X 4
		SHRADCODE = tf.einsum('mijkl,mjn->mikln',SHRAD,CODES)
		return SHRADCODE

	def AtomEmbToAtomEnergy(self,emb,Zs):
		"""
		Per-embedded vector, send each to
		an appropriate atom sub-net.
		"""
		# Step 1: Mol X MaxNAtom X nsh X nrad X ncode
		# => mol X
		#embshp = tf.shape(emb)
		#embdim = tf.reduce_prod(embshp[2:])
		embf = tf.reshape(emb,(self.batch_size*self.MaxNAtom,-1))
		Zrs = tf.cast(tf.reshape(Zs,(self.batch_size*self.MaxNAtom,-1)),self.prec)
		branches=[]
		for ele in self.AtomTypes:
			msk = tf.where(tf.equal(Zrs,ele),tf.ones_like(Zrs),tf.zeros_like(Zrs))
			l1 = tf.layers.dense(inputs=embf,units=256,activation=sftpluswparam,use_bias=True)
			l2 = tf.layers.dense(inputs=l1,units=256,activation=sftpluswparam,use_bias=True)
			l3 = tf.layers.dense(l2,units=1,activation=None,use_bias=True)
			branches.append(l3*msk)
		output = tf.reshape(tf.add_n(branches),(self.batch_size,self.MaxNAtom,1))
		return output

	def AtomEmbToAtomEnergyChannel(self,emb,Zs):
		"""
		A version without atom branches which instead uses the atom
		channels of its input.
		"""
		# Step 1: Mol X MaxNAtom X nsh X nrad X ncode
		# => mol X
		#embshp = tf.shape(emb)
		#embdim = tf.reduce_prod(embshp[2:])
		embf = tf.reshape(emb,(self.batch_size*self.MaxNAtom,-1))
		Zrs = tf.cast(tf.reshape(Zs,(self.batch_size*self.MaxNAtom,-1)),self.prec)
		# Gather the appropriate channels.
		tf.reshape(tf.gather_nd(elecode, Zrs),(self.batch_size,self.MaxNAtom,-1)) # mol X maxNatom X 4
		branches=[]
		for ele in self.AtomTypes:
			msk = tf.where(tf.equal(Zrs,ele),tf.ones_like(Zrs),tf.zeros_like(Zrs))
			l1 = tf.layers.dense(inputs=embf,units=256,activation=sftpluswparam,use_bias=True)
			l2 = tf.layers.dense(inputs=l1,units=256,activation=sftpluswparam,use_bias=True)
			l3 = tf.layers.dense(l2,units=1,activation=None,use_bias=True)
			branches.append(l3*msk)
		output = tf.reshape(tf.add_n(branches),(self.batch_size,self.MaxNAtom,1))
		return output

	def train_step(self,step):
		feed_dict = self.NextBatch(self.mset)
		_,train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
		self.print_training(step, train_loss)
		return

	def print_training(self, step, loss_):
		if (step%10==0):
			print("step: ", "%7d"%step, "  train loss: ", "%.10f"%(float(loss_)))
			print(self.sess.run([self.gp_tf])[0])
			feed_dict = self.NextBatch(self.mset)
			ens = self.sess.run([self.MolEnergies], feed_dict=feed_dict)
			for i in range(10):
				print("Pred, true: ", ens[0][i], feed_dict[self.groundTruthE_pl][i])
			print("Mean Abs Error: ", np.average(np.abs(ens[0]-feed_dict[self.groundTruthE_pl])))
			#print("RotGrad:",self.sess.run([self.RotGrad], feed_dict=feed_dict))
		return

	def training(self, loss):
		tf.summary.scalar(loss.op.name, loss)
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def Train(self,mxsteps=3000):
		test_freq = 40
		for step in range(1, mxsteps+1):
			self.train_step(step)
		return

	def Prepare(self):
		self.MaxZtf = tf.constant(MAX_ATOMIC_NUMBER)
		self.xyzs_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,3), dtype = self.prec)
		self.zs_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,1), dtype = tf.int32)

		self.groundTruthE_pl = tf.placeholder(shape = (self.batch_size,1), dtype = tf.float64)
		self.groundTruthG_pl = tf.placeholder(shape = (self.batch_size,self.MaxNAtom,3), dtype = tf.float64)

		self.atom_codes = tf.Variable(self.AtomCodes,trainable=True)
		self.gp_tf  = tf.Variable(self.GaussParams,trainable=True, dtype = self.prec)

		if 0:
			thetas = tf.acos(2.0*tf.random_uniform([self.batch_size],dtype=tf.float64)-1.0)
			phis = tf.random_uniform([self.batch_size],dtype=tf.float64)*2*Pi
			psis = tf.random_uniform([self.batch_size],dtype=tf.float64)*2*Pi
			matrices = TF_RotationBatch(thetas,phis,psis)
			self.xyzs_shifted = self.xyzs_pl - self.xyzs_pl[:,0,:][:,tf.newaxis,:]
			tmpxyzs = tf.einsum('ijk,ikl->ijl',self.xyzs_shifted, matrices)

		self.dxyzs = tf.expand_dims(self.xyzs_pl, axis=2) - tf.expand_dims(self.xyzs_pl, axis=1)
		self.z1z2 = tf.cast(tf.expand_dims(self.zs_pl, axis=2) * tf.expand_dims(self.zs_pl, axis=1),tf.float64)
		self.pair_mask = BuildMultiplicativeZeroMask(self.z1z2)
		# Canonicalized difference Vectors.
		self.cdxyzs = CanonicalizeGS(self.dxyzs*self.pair_mask)
		self.embedded = self.Embed(self.cdxyzs, self.zs_pl, self.pair_mask, self.gp_tf, self.atom_codes, self.l_max)
		self.AtomEnergies = self.AtomEmbToAtomEnergy(self.embedded,self.zs_pl)
		self.MolEnergies = tf.reduce_sum(self.AtomEnergies,axis=1,keepdims=False)

		# Optional. Verify that the canonicalized differences are invariant.
		#self.RotGrad = tf.gradients(self.embedded,psis)[0]

		self.loss = tf.losses.mean_squared_error(self.MolEnergies, self.groundTruthE_pl)
		tf.add_to_collection('losses', self.loss)
		self.train_op = self.training(self.loss)

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

net = InGauShBPNetwork(b)
net.Train()
