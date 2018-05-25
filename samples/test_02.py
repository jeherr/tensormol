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
import TensorMol
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

def TestTraining():
	b = MSet("HNCO_small")
	b.Load()
	b.cut_max_num_atoms(40)
	b.cut_max_grad(2.0)
	b.cut_energy_outliers()
	net = SparseCodedChargedGauSHNetwork(aset=b,load=False,load_averages=False,mode='train')
	net.Train()

def TestNeb():
	net = TensorMol.TFNetworks.SparseCodedChargedGauSHNetwork(aset=None,load=True,load_averages=True,mode='eval')
	s = TensorMol.MSet()
	s.ReadXYZ("Endiandric")
	EFt = net.GetBatchedEnergyForceRoutine(s)
	txyz = np.zeros((len(s.mols),s.mols[0].coords.shape[0],s.mols[0].coords.shape[1]))
	for i in range(len(s.mols)):
		txyz[i] = s.mols[i].coords
	NEB = TensorMol.Simulations.BatchedNudgedElasticBand(net,s.mols[0],s.mols[3],thresh_=0.003,nbeads_=20)
	NEB.Opt(eff_max_step=100)

def TestOpt():
	m = TensorMol.Containers.Mol()
	m.FromXYZString("""73

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
	net = TensorMol.TFNetworks.SparseCodedChargedGauSHNetwork(aset=None,load=True,load_averages=True,mode='eval')
	EF = net.GetEnergyForceRoutine(m)
	Opt = TensorMol.Simulations.GeomOptimizer(EF)
	m.Distort(0.25)
	m1=Opt.Opt(m,"TEST",eff_max_step=500)

TestOpt()
TestNeb()


#(self,bf_,g0_,g1_,name_="Neb",thresh_=None,nbeads_=None):
if 0:
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

if 0:
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
from TensorMol.Containers import Mol

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
