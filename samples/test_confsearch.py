
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
	sugarXYZ="""6

H          0.84754        0.03474        1.03453
C          0.35048        0.00675        0.06084
H          0.63507        0.89276       -0.52006
H          0.66294       -0.89338       -0.48283
O         -1.01083       -0.00823        0.36439
H         -1.48520       -0.03263       -0.45685
"""
	sugarXYZ4="""32
Comment: ;;;energy -5.428469095986943;;;OptStep 65
N   -6.26581092151765  0.10201704645376442  0.4844611063243624
C   -5.594603500859812  -0.1476439779696308  1.6540229580344854
H   -6.06408166187968  0.03395579772152954  2.611400946380539
N   -4.397108937400867  -0.608778588506067  1.4867020029477787
C   -4.251058537884246  -0.7006179699769643  0.1254846033786028
C   -3.1197020902842434  -1.1797064941566946  -0.6479754602396602
O   -2.0534884245256846  -1.6020656842090555  -0.2770964360818963
N   -3.426318161751284  -1.0985810157661748  -2.0247102409023254
H   -2.6894394606334497  -1.4634499934316327  -2.620839083514288
C   -4.5844941918117765  -0.6027566125663602  -2.5593703194183144
N   -4.680800906702206  -0.6067687769342851  -3.914462143891655
H   -3.8643811711725338  -0.7254267122491926  -4.489224175144041
H   -5.460642737665189  -0.10958517740122226  -4.312748373466173
N   -5.593842069060168  -0.15808626335257103  -1.8499503366185226
C   -5.389039819794843  -0.2654878865803788  -0.5283051683486195
C   -7.6835635133596325  0.4080953120877223  0.37178264899690716
C   -7.991712123123099  1.5266942559287053  -0.616278732051835
O   -8.394057424093468  -0.7266900650242054  -0.08908220682351803
H   -8.006629939149892  0.6422733679885222  1.3859661485264032
C   -9.396075658454809  1.1908739101130064  -1.1128084695419624
H   -7.301458525626736  1.435191326026471  -1.454035684455011
C   -9.463315583356628  -0.3347830905337629  -0.941720815440403
O   -10.42156440916291  1.7335645130474004  -0.30250958924104787
H   -9.538463910658399  1.486084005350539  -2.1582976979507817
C   -9.314499180151143  -1.089758335160013  -2.2464311193635975
H   -10.409198385162654  -0.5828912814056532  -0.4528641294875817
H   -10.471629815288331  2.6796046083944605  -0.44371486922132697
O   -10.287860131435679  -0.7324197543787787  -3.202859305621027
H   -8.33084039738857  -0.8623775609959942  -2.6656772660556074
H   -9.345746729570756  -2.159580552631347  -2.019300956043635
H   -11.153038334247686  -0.9091693290741931  -2.830985204701773
H   -7.898343346825952  2.5172109791920496  -0.16913263096446932
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

	sugarXYZ3 = """64

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
	Opt = GeomOptimizer(F, efh_=EFH)
	molecule = Opt.Opt(m, eff_thresh=0.0002)
	# Gotta optimize before running spectra
	w, v, i, TD = HarmonicSpectra(
		lambda x: F(x,False),
		m.coords,
		m.atoms,
		WriteNM_=True,
		Mu_=DipoleField,
		h_ = lambda x: EFH(x)[2])
	print(molecule, w, i, TD)
	exit(0)

	MOpt = ScannedOptimization(F,m)
	m = MOpt.Search(m)
	exit(0)

#Eval()
#TestBetaHairpin()
#TestUrey()
#HarmonicSpectra()
ConfSearch()
