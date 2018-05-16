"""
This revision shares low-layers to avoid overfitting, and save time.
It's a little recurrent-y.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
HAS_MATPLOTLIB=False
if (0):
	try:
		import matplotlib.pyplot as plt
		HAS_MATPLOTLIB=True
	except Exception as Ex:
		HAS_MATPLOTLIB=False

from TensorMol import *
import numpy as np

if (0):
	a = MSet("Heavy")
	b = MSet("CrCuSiBe")
	c = MSet("CrCu")
	d = MSet("MHMO_withcharge")
	e = MSet("kevin_heteroatom.dat")
	f = MSet("HNCO_small")
	g = MSet("chemspider20_1_meta_withcharge_noerror_all")
	sets = [a,b,c,d,e,f,g]
	S = MSet()
	for s in sets:
		s.Load()
		S.mols = S.mols + s.mols
	S.cut_max_grad(2.)
	S.cut_max_atomic_number(37)
	S.cut_max_num_atoms(30)
	S.cut_energy_outliers(2.)
	S.Save("PeriodicTable")

if (0):
	a = MSet("chemspider20_345_opt")
	b = MSet("chemspider20_1_opt_withcharge_noerror_part2_max50")
	c = MSet("chemspider20_1_meta_withcharge_noerror_all")
	d = MSet("kevin_heteroatom.dat")
	e = MSet("chemspider20_24578_opt")
	g = MSet("KevinHeavy")
	sets = [g]
	sets[-1].Load()
	f = MSet("chemspider12_clean_maxatom35")
	f.Load()
	while (len(f.mols)>0):
		sets.append(MSet())
		while(len(f.mols)>0 and len(sets[-1].mols)<100000):
			sets[-1].mols.append(f.mols.pop())
	UniqueSet = MSet("UniqueBonds")
	UniqueSet.Load()
	MasterSet = MSet("MasterSet")
	MasterSet.Load()
	for aset in sets:
		for i,amol in enumerate(aset.mols):
			amol.GenSummary()
			if i%10000 == 0:
				print(i)
		MasterSet.mols = MasterSet.mols+aset.mols
		aset.cut_unique_bond_hash()
		UniqueSet.mols = UniqueSet.mols+aset.mols
		UniqueSet.Save("UniqueBonds")
		MasterSet.Save("MasterSet")

if 0:
	b = MSet("MasterSet40")
	b.Load()
	b.cut_max_num_atoms(40)
	b.cut_max_grad(2.0)
	b.cut_energy_outliers()
	b.cut_max_atomic_number(37)
	#b.Save("MasterSet40")

if 0:
	b = MSet("HNCO_small")
	b.Load()
	b.cut_max_num_atoms(40)
	b.cut_max_grad(2.0)
	b.cut_energy_outliers()

if 1:
	b=MSet()
	m4 = Mol()
	m4.FromXYZString("""73

	O          3.11000        4.22490       -0.75810
	O          4.72290        2.06780       -2.02160
	O          3.28790        0.27660       -2.12830
	O          7.57740       -2.18410        0.83530
	O          6.93870       -0.24500        1.85400
	N         -0.44900        1.57680        0.54520
	N          0.67240       -1.09000        0.16920
	N         -3.08650        0.73580        0.30880
	N         -2.08930       -2.17120        0.36140
	C          3.15530        1.80910       -0.26920
	C          1.94310        0.99350        0.14610
	C          1.10470        3.11900       -0.07220
	C          0.85730        1.76100        0.25120
	C          2.53600        3.20940       -0.40610
	C         -0.01170        3.83890       -0.00490
	C          1.80260       -0.45930        0.35870
	C         -0.97430        2.76340        0.37660
	C          2.91740       -1.33270        0.77810
	C          2.32400       -2.53760        0.79670
	C          3.70160        1.28460       -1.55560
	C          0.92000       -2.41280        0.43150
	C         -2.41080        3.07580        0.55540
	C         -0.29110        5.26060       -0.27150
	C         -3.30810        2.07470        0.53020
	C         -4.22430       -0.01890        0.37480
	C         -1.33840       -3.26350       -0.02560
	C          0.02500       -3.41140        0.34800
	C         -4.76240        2.20220        0.74210
	C         -5.29980        0.95570        0.65430
	C         -3.38890       -2.29630       -0.08630
	C         -2.18770       -4.11130       -0.70980
	C         -3.46520       -3.50710       -0.75000
	C          4.24800       -0.96910        1.13640
	C         -4.40070       -1.34110        0.20870
	C          2.93270       -3.85050        1.16890
	C         -6.72700        0.53540        0.79750
	C         -1.82240       -5.42330       -1.29120
	C         -5.50430        3.40910        1.00050
	C         -4.63100       -4.03780       -1.35240
	C          5.32530       -1.71620        0.83820
	C          5.31710        1.65030       -3.25560
	C         -6.03270        4.16680        0.03880
	C         -5.68440       -3.34470       -1.89440
	C          6.67040       -1.26750        1.24620
	H          3.91490        1.82320        0.51520
	H         -2.69930        4.10740        0.71860
	H         -0.66660        5.75450        0.62990
	H          0.61110        5.79090       -0.59250
	H         -1.03960        5.36580       -1.06280
	H          0.35400       -4.43150        0.53040
	H         -5.40880       -1.73500        0.30730
	H          4.36030       -0.04870        1.70090
	H          3.88280       -3.75330        1.69880
	H          2.27760       -4.40220        1.85250
	H          3.09460       -4.46420        0.27690
	H         -6.83900       -0.17840        1.62010
	H         -7.08680        0.06850       -0.12540
	H         -7.38530        1.38280        1.01220
	H         -2.13640       -6.23380       -0.62560
	H         -2.30250       -5.57200       -2.26410
	H         -0.74330       -5.51440       -1.45110
	H         -5.62320        3.69670        2.04090
	H         -4.70070       -5.12240       -1.41710
	H          5.25800       -2.63030        0.25800
	H          4.57150        1.65850       -4.05620
	H          5.75710        0.65460       -3.14550
	H          6.11050        2.35890       -3.50730
	H         -6.58170        5.06770        0.29090
	H         -5.93330        3.91260       -1.01130
	H         -6.51300       -3.89300       -2.33170
	H         -5.71990       -2.26400       -1.94770
	H          8.49250       -1.93230        1.08330
	Mg        -1.34673        0.02041       -0.06327
	""")
	#b.mols.append(m4)
	m=Mol()
	m.FromXYZString("""10

	C         -2.10724        0.51908       -0.00001
	C         -1.49330       -0.51788        0.00002
	H         -2.60197        1.45402        0.00000
	C          1.78974        0.78601        0.00001
	C          1.32265       -0.32486       -0.00003
	H          2.15290        1.77949       -0.00001
	C         -0.66955       -1.70685       -0.00000
	C          0.66874       -1.61509        0.00001
	H         -1.14475       -2.68318        0.00001
	H          1.27274       -2.51742        0.00000
	""")
	#b.mols.append(m)
	m2=Mol()
	m2.FromXYZString("""5

	C          1.62942        0.65477        0.00000
	H          2.69942        0.65477        0.00000
	H          1.27276        1.65379        0.14017
	H          1.27275        0.03387        0.79509
	H          1.27275        0.27665       -0.93526
	""")
	b.mols.append(m2)
	m3=Mol()
	m3.FromXYZString("""17

	C         -0.49579        0.02905       -0.16610
	C          0.63324       -0.60448       -1.00024
	H          1.45203       -0.96642       -0.34138
	H          1.05971        0.13678       -1.71028
	H          0.25074       -1.46793       -1.58647
	C         -1.06604       -1.01999        0.80663
	H         -1.47567       -1.89007        0.24925
	H         -1.88382       -0.58297        1.41962
	H         -0.27438       -1.38856        1.49434
	C         -1.61377        0.52024       -1.10455
	H         -2.03216       -0.32525       -1.69245
	H         -1.22318        1.27947       -1.81627
	H         -2.44030        0.98186       -0.52208
	C          0.06341        1.22042        0.63377
	H          0.87311        0.88762        1.31872
	H         -0.73634        1.69321        1.24400
	H          0.48078        1.99083       -0.05018
	""")
	#b.mols.append(m3)

MAX_ATOMIC_NUMBER = 55

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
		self.learning_rate = 0.0005
		self.ncan = 12
		self.DoHess=False
		self.mode = mode
		self.Lsq = False
		if (mode == 'eval'):
			self.ncan = 12
		self.RCut_Coulomb = 19.0
		self.RCut_NN = 7.0
		self.AtomCodes = ELEMENTCODES
		#self.AtomCodes = np.random.random(size=(MAX_ATOMIC_NUMBER,6))
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
		self.AverageElementEnergy = np.zeros((MAX_ATOMIC_NUMBER))
		self.AverageElementCharge = np.zeros((MAX_ATOMIC_NUMBER))
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
		nlt_nn, nlt_j = self.NLTensors(xyzs_t,Zs_t)
		if ((self.MaxNeigh_J > self.MaxNeigh_J_prep) or (self.MaxNeigh_NN > self.MaxNeigh_NN_prep)):
			self.batch_size=1
			MustPrepare=True
		if (MustPrepare):
			self.Prepare()
			self.Load()
		self.CAxes = np.zeros((self.ncan,self.batch_size*self.MaxNAtom,3,3))
		self.CWeights = np.zeros((self.ncan,self.batch_size,self.MaxNAtom))
		self.CWeights, self.CAxes = self.sess.run([self.CanonicalAxes(self.dxyzs,self.sparse_mask)],feed_dict=self.MakeFeed(Mol(m.atoms,m.coords)))[0]
		def EF(xyz_,DoForce=True,Debug = False):
			feed_dict = self.MakeFeed(Mol(m.atoms,xyz_))
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
		MustPrepare=True
		if (mset.MaxNAtom() > self.MaxNAtom):
			self.MaxNAtom = mset.MaxNAtom()
		xyzs_t = np.zeros((self.batch_size,self.MaxNAtom,3))
		Zs_t = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
		xyzs_t[0,:m.NAtoms(),:] = m.coords
		Zs_t[0,:m.NAtoms(),0] = m.atoms
		nlt_nn, nlt_j = self.NLTensors(xyzs_t,Zs_t)
		if ((self.MaxNeigh_J > self.MaxNeigh_J_prep) or (self.MaxNeigh_NN > self.MaxNeigh_NN_prep)):
			self.batch_size=1
			MustPrepare=True
		if (MustPrepare):
			self.Prepare()
			self.Load()
		def EF(xyz_,DoForce=True,Debug = False):
			xyzs = np.zeros((self.batch_size,self.MaxNAtom,3))
			Zs = np.zeros((self.batch_size,self.MaxNAtom,1),dtype=np.int32)
			nls_nn = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_NN),dtype=np.int32)
			nls_j = -1*np.ones((self.batch_size,self.MaxNAtom,self.MaxNeigh_J),dtype=np.int32)
			xyzs[0,:m.NAtoms(),:] = xyz_
			Zs[0,:m.NAtoms(),0] = m.atoms
			nlt_nn, nlt_j = self.NLTensors(xyzs,Zs)
			if ((self.MaxNeigh_J > self.MaxNeigh_J_prep) or (self.MaxNeigh_NN > self.MaxNeigh_NN_prep)):
				print("Too Many Neighbors.")
				raise Exception('NeighborOverflow')
			nls_nn[:nlt_nn.shape[0],:nlt_nn.shape[1],:nlt_nn.shape[2]] = nlt_nn
			nls_j[:nlt_j.shape[0],:nlt_j.shape[1],:nlt_j.shape[2]] = nlt_j
			feed_dict = {self.xyzs_pl:xyzs, self.zs_pl:Zs,self.nl_nn_pl:nls_nn,self.nl_j_pl:nls_j}
			if (self.DoRotGrad):
				print("RotGrad:",self.sess.run([self.RotGrad], feed_dict=feed_dict))
			if (Debug):
				print(nls_nn,nls_j)
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
			print("Too Many Neighbors.")
			raise Exception('NeighborOverflow')
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
			print("Too Many Neighbors.")
			raise Exception('NeighborOverflow')
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
		if (1):
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

			v3refl = tf.cross(v1n,v2n)
			posz = tf.tile(tf.greater(tf.reduce_sum(v3refl*tf.constant([[1.,1.,1.]],dtype=self.prec),keepdims=True,axis=-1),0.),[1,3])
			v3 = tf.where(posz,v3refl,-1.*v3refl)
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
			ca = tf.where(tf.greater(w3p,0.95),tf.zeros_like(w3p),tf.cos(w3p/0.95*Pi/2.))

			Cutoffs = cw1*cw2*ca*msk[:,perm[0],:]*msk[:,perm[1],:]
			weightstore.append(w1+w2)
			cuttore.append(Cutoffs)
			axtore.append(vs)

			if 1:
				vs2 = tf.concat([second[:,tf.newaxis,:],first[:,tf.newaxis,:],-1*v3[:,tf.newaxis,:]],axis=1)
				weightstore.append(w1+w2)
				cuttore.append(Cutoffs)
				axtore.append(vs)


		Cuts = tf.stack(cuttore,axis=0)
		pw = Cuts*tf.exp(-tf.stack(weightstore,axis=0))#/(tf.stack(weightstore,axis=0) + 1e-9)
		tdn = tf.where(tf.greater_equal(Cuts,0.), pw, tf.zeros_like(pw))
		dn = tf.reduce_sum(tdn,axis=0,keepdims=True)
		tw = tf.where(tf.greater_equal(Cuts,0.), tdn/(dn+1e-19), tf.zeros_like(tdn))
		weights = tf.reshape(tw,(self.ncan,argshape[0],argshape[1]))

		return weights, tf.stack(axtore,axis=0)

	def CanonicalizeAngleAverage(self,dxyzs,sparse_mask):
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
		if (0):
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
		tore = []
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

			v3refl = tf.cross(v1n,v2n)
			posz = tf.tile(tf.greater(tf.reduce_sum(v3refl*tf.constant([[1.,1.,1.]],dtype=self.prec),keepdims=True,axis=-1),0.),[1,3])
			v3 = tf.where(posz,v3refl,-1.*v3refl)
			v3 *= safe_inv_norm(v3)

			# Compute the average of v1, v2, and their projections onto the plane.
			v_av = (v1n+v2n)/2.0
			v_av *= safe_inv_norm(v_av)

			# Rotate pi/4 cw and ccw to obtain v1,v2
			first = TF_AxisAngleRotation(v3,v_av,tf.constant(Pi/4.,dtype=self.prec))
			second = TF_AxisAngleRotation(v3,v_av,tf.constant(-Pi/4.,dtype=self.prec))

			vs = tf.concat([first[:,tf.newaxis,:],second[:,tf.newaxis,:],v3[:,tf.newaxis,:]],axis=1)
			tore.append(tf.reshape(tf.einsum('ijk,ilk->ijl',realdata,vs),tf.shape(dxyzs)))

			cw1 = tf.where(tf.less(w1,axis_cutoff),tf.cos(w1/axis_cutoff*Pi/2.0),tf.zeros_like(w1))
			cw2 = tf.where(tf.less(w2,axis_cutoff),tf.cos(w2/axis_cutoff*Pi/2.0),tf.zeros_like(w2))
			ca = tf.where(tf.greater(w3p,0.95),tf.zeros_like(w3p),tf.cos(w3p/0.95*Pi/2.))

			Cutoffs = cw1*cw2*ca*msk[:,perm[0],:]*msk[:,perm[1],:]
			weightstore.append(w1+w2)
			cuttore.append(Cutoffs)
			axtore.append(vs)

			if 0:
				vs2 = tf.concat([second[:,tf.newaxis,:],first[:,tf.newaxis,:],-1*v3[:,tf.newaxis,:]],axis=1)
				tore.append(tf.reshape(tf.einsum('ijk,ilk->ijl',realdata,vs2),tf.shape(dxyzs)))
				weightstore.append(w1+w2)
				cuttore.append(Cutoffs)
				axtore.append(vs)


		tformedcoords = tf.stack(tore,axis=0)
		Cuts = tf.stack(cuttore,axis=0)
		#pw = tf.nn.softmax(-1*tf.stack(weightstore,axis=0),axis=0)*Cuts
		pw = Cuts*tf.exp(-tf.stack(weightstore,axis=0))#/(tf.stack(weightstore,axis=0) + 1e-9)
		tdn = tf.where(tf.greater_equal(Cuts,0.), pw, tf.zeros_like(pw))
		dn = tf.reduce_sum(tdn,axis=0,keepdims=True)
		tw = tf.where(tf.greater_equal(Cuts,0.), tdn/(dn+1e-19), tf.zeros_like(tdn))

		weights = tf.reshape(tw,(self.ncan,argshape[0],argshape[1]))
		#weights = tf.Print(weights,[weights[:,0,:2]],"Weights",summarize=100000)
		#weights = tf.Print(weights,[tf.reduce_sum(tf.reduce_mean(tformedcoords,axis=3)*weights[...,tf.newaxis],axis=0)],"Center of Canonicalization",summarize=100000)
		#weights = tf.Print(weights,[self.nl_nn_pl[0,:2]],"NL",summarize=10000)
		#weights = tf.Print(weights,[weights[:,0,:2]],"Weights",summarize=10000)
		return tformedcoords, weights, tf.stack(axtore,axis=0)

	def CanonicalizeGS(self,dxyzs,sparse_mask):
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
		if (self.ncan == 1):
			orders = [[0,1]]
		if (self.ncan == 2):
			orders = [[0,1],[1,0]]
		elif (self.ncan == 6):
			orders = [[0,1],[1,0],[1,2],[2,1],[0,2],[2,0]]
		elif (self.ncan == 12):
			orders = [[0,1],[1,0],[1,2],[2,1],[0,2],[2,0],[0,3],[3,0],[1,3],[3,1],[3,2],[2,3]]
		elif (self.ncan == 16):
			orders = [[0,1],[1,0],[1,2],[2,1],[0,2],[2,0],[0,3],[3,0],[1,3],[3,1],[3,2],[2,3],[0,4],[4,0],[1,4],[4,1]]
		elif (self.ncan == 20):
			orders = [[0,1],[1,0],[1,2],[2,1],[0,2],[2,0],[0,3],[3,0],[1,3],[3,1],[3,2],[2,3],[0,4],[4,0],[1,4],[4,1],[2,4],[4,2],[3,4],[4,3]]
		tore = []
		weightstore = []
		for perm in orders:
			v1 = tf.reshape(dxyzs[:,:,perm[0],:],(argshape[0]*argshape[1],3))+tf.constant(np.array([1e-8,0.,0.]),dtype=tf.float64)
			w1 = (tf.reshape(tf.reduce_sum(dxyzs[:,:,perm[0],:]*dxyzs[:,:,perm[0],:],axis=-1),(argshape[0]*argshape[1],1))+1e-8)
			v1 *= safe_inv_norm(v1)
			v2 = tf.reshape(dxyzs[:,:,perm[1],:],(argshape[0]*argshape[1],3))+tf.constant(np.array([0.,1e-8,0.]),dtype=tf.float64)
			#w2 = (tf.reshape(tf.reduce_sum(dxyzs[:,:,perm[1],:]*dxyzs[:,:,perm[1],:],axis=-1),(argshape[0]*argshape[1],1))+1e-8)
			v2 -= tf.einsum('ij,ij->i',v1,v2)[:,tf.newaxis]*v1
			v2 *= safe_inv_norm(v2)
			v3refl = tf.cross(v1,v2)
			posz = tf.tile(tf.greater(tf.reduce_sum(v3refl*tf.constant([[1.,1.,1.]],dtype=self.prec),keepdims=True,axis=-1),0.),[1,3])
			v3 = tf.where(posz,v3refl,-1*v3refl)
			v3 *= safe_inv_norm(v3)
			vs = tf.concat([v1[:,tf.newaxis,:],v2[:,tf.newaxis,:],v3[:,tf.newaxis,:]],axis=1)
			tore.append(tf.reshape(tf.einsum('ijk,ilk->ijl',realdata,vs),tf.shape(dxyzs)))
			#weightstore.append((w1+w2)*msk[:,perm[0],:]*msk[:,perm[1],:])
			weightstore.append((w1)*msk[:,perm[0],:]*msk[:,perm[1],:])
		return tf.stack(tore,axis=0), tf.reshape(tf.nn.softmax(-1*tf.stack(weightstore,axis=0),axis=0),(self.ncan,argshape[0],argshape[1]))

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
			stdinit = tf.constant(np.ones(MAX_ATOMIC_NUMBER),dtype=self.prec)
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
			stdinit = tf.constant(np.ones(MAX_ATOMIC_NUMBER),dtype=self.prec)
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
			GD = self.sess.run([self.GradDiff], feed_dict=feed_dict)[0]
			print("Grad Diff:", np.reduce_sum(GD,axis=(-1,-2)))
			for k,mol in enumerate(mols):
				print(k,mol)
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
			if (self.mode=='train'):
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

#net = SparseCodedChargedGauSHNetwork(aset=b,load=False,load_averages=False,mode='train')
#net.Train()
net = SparseCodedChargedGauSHNetwork(aset=None,load=True,load_averages=True,mode='eval')

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

if 1:
	for i in range(5):
		mi = np.random.randint(len(b.mols))
		m = copy.deepcopy(b.mols[mi])
		print(m.atoms, m.coords)
		EF = net.GetEnergyForceRoutine(m)
		print(EF(m.coords))
		Opt = GeomOptimizer(EF)
		m1=Opt.Opt(m,"TEST"+str(i),eff_max_step=100)
		#m2 = Mol(m.atoms,m.coords)
		#m2.Distort(0.3)
		#m2=Opt.Opt(m2,"FromDistorted"+str(i))
		# Do a detailed energy, force scan for geoms along the opt coordinate.
		interp = m1.Interpolation(b.mols[mi],n=20)
		ens = np.zeros(len(interp))
		fs = np.zeros((len(interp),m1.NAtoms(),3))
		axes = []
		ws = np.zeros((len(interp),m1.NAtoms(),net.ncan))
		ews = np.zeros((len(interp),m1.NAtoms(),net.ncan))
		xws = np.zeros((len(interp),m1.NAtoms(),net.ncan))
		yws = np.zeros((len(interp),m1.NAtoms(),net.ncan))
		zws = np.zeros((len(interp),m1.NAtoms(),net.ncan))
		Xc = []
		Yc = []
		Zc = []
		Xa = []
		Ya = []
		Za = []
		for j,mp in enumerate(interp):
			ens[j],fs[j] = EF(mp.coords)
			ate = net.sess.run([net.CAEs[:,:,:,0]],feed_dict=net.MakeFeed(mp))[0]
			#a,bp,axs = net.sess.run([net.CanonicalizeGS(net.dxyzs)],feed_dict=net.MakeFeed(mp))[0]
			cdyx, bp, axs = net.sess.run([net.CanonicalizeAngleAverage(net.dxyzs,net.sparse_mask)],feed_dict=net.MakeFeed(mp))[0]
			#print(a,"b",b)
			ws[j]=np.transpose(bp[:,0,:m1.NAtoms()])
			if (j>0):
				sw1 = np.sort(ws[j])
				sw2 = np.sort(ws[j-1])
				if (np.any(np.greater(np.abs(sw1-sw2),0.1))):
					print("Discontinuity:", j)
					print("MOL1",interp[j-1])
					print("MOL2",interp[j])
					pblmaxes = np.stack(np.where(np.greater(np.abs(ws[j]-ws[j-1]),0.1)),axis=-1)
					print(pblmaxes)
					Inp1 = net.MakeFeed(interp[j-1])
					Inp2 = net.MakeFeed(interp[j])
					for pb in pblmaxes:
						print("Issue: ", pb)
						pbatom = pb[0]
						print("Wj-1",ws[j-1][pb[0]])
						print("Wj",ws[j][pb[0]])
						NL1 = Inp1[net.nl_nn_pl][0,pbatom]
						NL2 = Inp2[net.nl_nn_pl][0,pbatom]
						print("j-1nl",NL1)
						print("jnl",NL2)
						# Construct the distance matrices.
						x1 = Inp1[net.xyzs_pl][0]
						x2 = Inp2[net.xyzs_pl][0]
						print(x1,x2)
						nc1 = x1[NL1[np.where(NL1>0)]]
						nc2 = x2[NL2[np.where(NL2>0)]]
						print(nc1,nc2)
						d1s = np.linalg.norm(nc1 - (x1[pbatom])[np.newaxis,...],axis=-1)
						d2s = np.linalg.norm(nc2 - (x2[pbatom])[np.newaxis,...],axis=-1)
						print("Dists: ",d1s)
						print("Dists: ",d2s)

			ews[j]=np.transpose(ate[:,0,:m1.NAtoms()])
			if 1:
				axs2 = np.reshape(axs,(net.ncan,net.batch_size,net.MaxNAtom,3,3))
				xws[j]=np.transpose(axs2[:,0,:m1.NAtoms(),0,0])
				yws[j]=np.transpose(axs2[:,0,:m1.NAtoms(),1,0])
				zws[j]=np.transpose(axs2[:,0,:m1.NAtoms(),2,0])
		import matplotlib.pyplot as plt
		#for k in range(m1.NAtoms()):
		plt.plot(np.reshape(ws,(len(interp),m1.NAtoms()*net.ncan)))
		plt.show()
		plt.plot(np.reshape(ews,(len(interp),m1.NAtoms()*net.ncan)))
		plt.show()
		if (1):
			plt.plot(np.reshape(xws,(len(interp),m1.NAtoms()*net.ncan)))
			plt.show()
			plt.plot(np.reshape(yws,(len(interp),m1.NAtoms()*net.ncan)))
			plt.show()
			plt.plot(np.reshape(zws,(len(interp),m1.NAtoms()*net.ncan)))
			plt.show()
		plt.plot(ens)
		plt.show()
		# Compare force projected on each step with evaluated force.
		#for j in range(1,len(interp)):
		#	fs[j-1]

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

m = Mol()
m.FromXYZString("""68

Ti        0.120990   -0.060138    0.681291
 O        -1.183156   -1.211327    1.530892
 O         1.551867   -0.526524    1.879088
 O        -0.393679    1.496824    1.600554
 N        -1.573945   -0.081347   -0.721848
 N         1.163022   -1.708772   -0.348239
 N         1.168753    1.447422   -0.369693
 N         0.487855   -0.147761   -2.466560
 C        -2.365392   -1.436265    1.003740
 C        -2.630885   -0.805726   -0.264150
 C        -3.871117   -1.056995   -0.909339
 H        -4.062406   -0.639027   -1.891098
 C        -4.810176   -1.864998   -0.296374
 H        -5.754316   -2.064232   -0.795191
 C        -4.551256   -2.451304    0.963384
 H        -5.308554   -3.077665    1.426832
 C        -3.338502   -2.249950    1.606902
 H        -3.114836   -2.706156    2.565852
 C         2.390952   -1.504643    1.624142
 C         2.203137   -2.202060    0.378359
 C         3.118718   -3.228959    0.026910
 H         3.033398   -3.722479   -0.934718
 C         4.139844   -3.563900    0.896557
 H         4.845856   -4.341553    0.619056
 C         4.291634   -2.894854    2.132633
 H         5.100658   -3.180429    2.799223
 C         3.432176   -1.868643    2.494264
 H         3.541561   -1.328464    3.429274
 C         0.082181    2.697242    1.272502
 C         0.969870    2.709072    0.156623
 C         1.463287    3.948981   -0.298909
 H         2.098097    3.993581   -1.176906
 C         1.111612    5.117094    0.369597
 H         1.493951    6.069135    0.011693
 C         0.265473    5.084805    1.490221
 H         0.009661    6.008532    2.001673
 C        -0.258161    3.875463    1.941611
 H        -0.927590    3.822434    2.794486
 C        -0.559135    0.775222   -2.717545
 C        -1.673529    0.742001   -1.844994
 C        -2.747040    1.620694   -2.088361
 H        -3.588610    1.633646   -1.403975
 C        -2.688343    2.527350   -3.142015
 H        -3.512568    3.217141   -3.299816
 C        -1.563565    2.579248   -3.973513
 H        -1.519560    3.293873   -4.790475
 C        -0.499053    1.706764   -3.757774
 H         0.379719    1.730920   -4.394999
 C         0.236139   -1.539331   -2.555194
 C         0.676848   -2.356131   -1.483516
 C         0.474604   -3.748583   -1.571271
 H         0.774452   -4.378943   -0.741040
 C        -0.187066   -4.298804   -2.663753
 H        -0.362641   -5.370480   -2.698613
 C        -0.656316   -3.477134   -3.695516
 H        -1.178183   -3.909357   -4.544504
 C        -0.445910   -2.101273   -3.639011
 H        -0.794965   -1.452681   -4.437269
 C         1.826934    0.329568   -2.399929
 C         2.143015    1.220203   -1.346856
 C         3.452548    1.735349   -1.286446
 H         3.720882    2.395684   -0.468840
 C         4.411945    1.346198   -2.216853
 H         5.422558    1.737580   -2.135304
 C         4.094111    0.436857   -3.231447
 H         4.846701    0.135097   -3.954237
 C         2.800449   -0.073173   -3.318151
 H         2.528822   -0.772568   -4.103764""")
EF = net.GetEnergyForceRoutine(m)
print(EF(m.coords))
Opt = GeomOptimizer(EF)
m=Opt.OptGD(m)
m.Distort(0.2)
m=Opt.OptGD(m,"FromDistorted")
