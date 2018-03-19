
from __future__ import absolute_import
from __future__ import print_function
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

def GetChemSpider12(a):
	TreatedAtoms = np.array([1,6,7,8], dtype=np.uint8)
	PARAMS["NetNameSuffix"] = "act_sigmoid100"
	PARAMS["learning_rate"] = 0.00001
	PARAMS["momentum"] = 0.95
	PARAMS["max_steps"] = 21
	PARAMS["batch_size"] =  50   # 40 the max min-batch size it can go without memory error for training
	PARAMS["test_freq"] = 1
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["EnergyScalar"] = 1.0
	PARAMS["GradScalar"] = 1.0/20.0
	PARAMS["DipoleScaler"]=1.0
	PARAMS["NeuronType"] = "sigmoid_with_param"
	PARAMS["sigmoid_alpha"] = 100.0
	PARAMS["HiddenLayers"] = [2000, 2000, 2000]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	#PARAMS["Erf_Width"] = 1.0
	#PARAMS["Poly_Width"] = 4.6
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	#PARAMS["AN1_r_Rc"] = 8.0
	#PARAMS["AN1_num_r_Rs"] = 64
	PARAMS["EECutoffOff"] = 15.0
	#PARAMS["DSFAlpha"] = 0.18
	PARAMS["DSFAlpha"] = 0.18
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]
	#PARAMS["KeepProb"] = 0.7
	PARAMS["learning_rate_dipole"] = 0.0001
	PARAMS["learning_rate_energy"] = 0.00001
	PARAMS["SwitchEpoch"] = 2
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("Mol_chemspider12_maxatom35_H2O_with_CH4_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout_act_sigmoid100_rightalpha", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	return manager

def ConfSearch():
	"""
	Gather statistics about the metadynamics exploration process varying bump depth, and width.
	"""
	sugarXYZ2="""32

N   -6.3271259213339945  -0.1680050055012942  0.5494821438522681
C   -5.534774555224372  -0.9324366020933004  -0.25884062997924473
H   -5.855746299190421  -1.213688434637595  -1.253892543875676
N   -4.408122704903144  -1.2598137913065957  0.28420858305934926
C   -4.443643226222282  -0.7056911991264184  1.5397257234621518
C   -3.473370820868441  -0.753471646781067  2.622293517869121
O   -2.3967098012560935  -1.2887351697281098  2.671801453314543
N   -3.9733390480660438  -0.031189936663405415  3.7338327413657266
H   -3.3300112663045276  0.0013776154594821717  4.518165820650112
C   -5.189839898242146  0.5978117060702232  3.8025913907642885
N   -5.468625990713388  1.2598430842437982  4.958229161779782
H   -4.986990550136317  1.009062778014115  5.805778091676214
H   -6.416830816887925  1.585655962285715  5.060253134691255
N   -6.059493010673389  0.6325006926305676  2.8221032278957994
C   -5.627282516460912  -0.02343628481351278  1.7351124469676513
C   -7.688470037685264  0.2986213994383241  0.33838989151260346
C   -7.807432814472307  1.3644549656621816  -0.739421337466108
O   -8.485775780624218  -0.7749455092032562  -0.11077447202348721
H   -8.0352871966501  0.6594688674522282  1.3047478483959036
C   -9.246390256805395  1.1998363161286454  -1.2259897321436477
H   -7.123895789982109  1.107479062924131  -1.552837160198834
C   -9.531478666333477  -0.28828895514806385  -0.9429947973010968
O   -10.177020941734305  1.941390643356891  -0.4646020597443881
H   -9.34650264259604  1.4396969743287915  -2.2915622721999647
C   -9.563619056969989  -1.1471087679789878  -2.1922282608324832
H   -10.479161977373952  -0.35658690067487  -0.40212149079618476
H   -10.031093422184632  2.8767540371168696  -0.6127418162627264
O   -10.585642524149915  -0.7671901755804165  -3.086516387692085
H   -8.60491724628771  -1.0354446582949972  -2.7097507606434745
H   -9.659410017787598  -2.190690086689475  -1.8746474256530017
H   -11.425210781684097  -1.0849166612922816  -2.749349199966277
H   -7.565000420195489  2.3644176804016857  -0.38262683047808854
	"""
	sugarXYZ2="""8

	H          1.18508       -0.00384        0.98752
	C          0.75162       -0.02244       -0.02084
	H          1.16693        0.83301       -0.56931
	H          1.11552       -0.93289       -0.51453
	C         -0.75159        0.02250        0.02089
	H         -1.16688       -0.83337        0.56870
	H         -1.11569        0.93261        0.51508
	H         -1.18499        0.00442       -0.98752
	"""

	sugarXYZ = """64

N         -3.89447        1.65309        6.81749
C         -2.57063        2.14154        6.45965
C         -1.64699        0.92451        6.28554
O         -2.02468       -0.23594        6.43716
H         -4.02517        0.66047        6.99292
H         -2.64502        2.66139        5.49749
C         -2.02267        3.08597        7.52310
H         -2.68640        3.94396        7.67151
H         -1.91915        2.57524        8.48701
H         -1.03546        3.46915        7.24394
H         -4.63267        2.31094        7.05109
N         -0.36118        1.10147        5.95240
C          0.51585       -0.05407        5.78730
C          1.90033        0.50416        5.41070
O          2.12615        1.70787        5.29510
C         -0.03556       -1.00004        4.71019
H          0.09332       -0.55285        3.71584
H         -1.11644       -1.12423        4.85290
C          0.61374       -2.38378        4.73229
H          0.59985       -2.79046        5.75103
H          1.66465       -2.31390        4.43234
C         -0.10349       -3.35227        3.79743
H         -0.08623       -2.98698        2.76557
H         -1.14504       -3.48009        4.10633
N          0.60027       -4.63404        3.83952
H          1.61585       -4.61210        3.79276
C          0.00821       -5.85250        3.91277
N          0.54731       -7.04107        3.94375
H          0.52772       -8.06905        3.98034
H          1.57072       -7.14339        3.91878
N         -1.22373       -6.15307        3.95198
H         -1.94081       -5.45147        3.92425
H         -1.46240       -7.13130        4.00859
H         -0.04851        1.98293        5.55257
H          0.58325       -0.56171        6.75685
N          2.92376       -0.33302        5.19316
C          4.24128        0.19822        4.83454
C          5.15702       -1.02910        4.66323
O          4.76033       -2.18459        4.81982
C          4.77514        1.12523        5.92614
H          4.00959        1.85567        6.21487
H          5.04163        0.55180        6.82249
C          5.98649        1.90536        5.47345
O          6.46754        1.85576        4.34671
N          6.55564        2.67538        6.44418
H          6.14712        2.79820        7.36086
H          7.35742        3.24084        6.19483
H          4.14553        0.71402        3.87166
H          2.87946       -1.31330        5.46211
N          6.44431       -0.86327        4.33009
C          7.31458       -2.02955        4.16752
C          8.70212       -1.47450        3.78999
O          8.93184       -0.27147        3.67303
C          6.78714       -2.96507        3.08046
C          7.38715       -4.33584        3.20115
O          8.28945       -4.68299        3.94650
O          6.84566       -5.20410        2.32571
H          7.38559       -2.52926        5.14115
H          7.01230       -2.57192        2.08255
H          5.70067       -3.07881        3.17123
H          7.30514       -6.05472        2.48689
H          6.79638        0.02969        3.99203
O          9.68926       -2.37377        3.58482
H          9.49060       -3.41415        3.68597
"""

	m = Mol()
	m.FromXYZString(sugarXYZ)

	#d,t,q = m.Topology()
	#print("Topology",d,t,q)

	#from MolEmb import EmptyInterfacedFunction, Make_NListNaive, Make_NListLinear
	#print("READ MOL XFOIUDOFIUDFO")
	#print(m.coords,15.0,m.NAtoms(),True)
	#EmptyInterfacedFunction(np.zeros((10,3)),13)
	#print("Passed test")
	#return

	s = MSet()
	s.mols.append(m)
	mgr = GetChemSpider12(s)
	def GetEnergyForceHess(m):
		def hess(x_):
			tmpm = Mol(m.atoms,x_)
			Energy, Force, Hessian = mgr.EvalBPDirectEEHessSingle(tmpm, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
			return Energy[0],Force[0],Hessian[0].reshape((3*m.NAtoms(),3*m.NAtoms()))
		return hess

	def GetChargeField(m):
		def Charges(x_):
			tmpm = Mol(m.atoms,x_)
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = mgr.EvalBPDirectEEUpdateSingle(tmpm, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			return atom_charge[0]
		return Charges

	def GetEnergyForceForMol(m):
		def EnAndForce(x_, DoForce=True):
			tmpm = Mol(m.atoms,x_)
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = mgr.EvalBPDirectEEUpdateSingle(tmpm, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			if DoForce:
				return energy, force
			else:
				return energy
		return EnAndForce

	# This actually affords a 2x speedup on set evaluation.
	def GetEnergyForceForSet(m,batch_size = 200):
		"""
		An evaluation routine for a batch of geometries to speed-up
		Hessian Evaluation.
		"""
		s = MSet()
		for i in range(batch_size):
			s.mols.append(m)
		manager = GetChemSpider12(s)
		def EnAndForceSet(s_, DoForce=True):
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSet(s_, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			if DoForce:
				return energy, force
			else:
				return energy
		return EnAndForceSet
	if 0:
		# Test the speed and accuracy of the set force evaluation.
		for i in range(200):
			F(m.coords)
		Fs = GetEnergyForceForSet(m)
		s = MSet()
		for i in range(200):
			s.mols.append(m)
		Fs(s)
		exit(0)

	EFH = GetEnergyForceHess(m)

	F = GetEnergyForceForMol(m)
	CF = GetChargeField(m)

	def DipoleField(x_):
		q = np.asarray(CF(x_))
		dipole = np.zeros(3)
		for i in range(0, q.shape[0]):
			dipole += q[i] * x_[i]
		return dipole

	PARAMS["OptMaxCycles"] = 2000
	PARAMS["OptThresh"] = 0.001

	# Test the hessian.
	Opt = GeomOptimizer(F,efh_=EFH)
	molecule = Opt.OptNewton(m)
	# Gotta optimize before running spectra
	w, v, i = HarmonicSpectra(
		lambda x: F(x,False),
		m.coords,
		m.atoms,
		WriteNM_=True,
		Mu_=DipoleField,
		h_ = lambda x: EFH(x)[2])
	return molecule, w, i, TD
	exit(0)

	MOpt = ScannedOptimization(F,m)
	m = MOpt.Search(m)
	exit(0)

#Eval()
#TestBetaHairpin()
#TestUrey()
#HarmonicSpectra()
ConfSearch()
