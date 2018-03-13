
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
	sugarXYZ=""" 32
Gaunosine-anti
N                 -6.31166300   -0.16201000    0.57617900
 C                 -5.51077700   -0.92984600   -0.25137800
 H                 -5.83876000   -1.21426600   -1.23858000
 N                 -4.37768700   -1.27744000    0.30205000
 C                 -4.42396100   -0.71740900    1.56655300
 C                 -3.48255600   -0.75638800    2.63564100
 O                 -2.37781700   -1.30038000    2.69618600
 N                 -3.96808100   -0.04709800    3.76106400
 H                 -3.32175800   -0.00958900    4.53941600
 C                 -5.18105700    0.59962700    3.84268300
 N                 -5.45302900    1.26792900    5.00066700
 H                 -5.01802200    0.97739200    5.86248100
 H                 -6.39833100    1.61231400    5.08295700
 N                 -6.04001200    0.63246300    2.85388000
 C                 -5.62094800   -0.02753800    1.75444500
 C                 -7.65708200    0.32695300    0.30518300
 C                 -7.77886700    1.35515400   -0.82248300
 O                 -8.44629500   -0.77495400   -0.12143500
 H                 -8.02583900    0.73332000    1.25070500
 C                 -9.25768400    1.20395800   -1.20980500
 H                 -7.12883000    1.07611100   -1.65725800
 C                 -9.52896100   -0.29114500   -0.94901400
 O                -10.10871400    1.95034400   -0.33980800
 H                 -9.45521700    1.45750300   -2.25386700
 C                 -9.57475400   -1.15120600   -2.21304900
 H                -10.46777000   -0.37993900   -0.39333600
 H                -10.17349800    2.85770800   -0.65427500
 O                -10.62686300   -0.75908900   -3.08122600
 H                 -8.64886000   -1.02448200   -2.78108500
 H                 -9.64721300   -2.20657000   -1.92764600
 H                -11.46843400   -1.07297200   -2.73426800
 H                 -7.52887600    2.36827700   -0.50575900
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

	def GetEnergyForceForMol(m):
		s = MSet()
		s.mols.append(m)
		manager = GetChemSpider12(s)
		def EnAndForce(x_, DoForce=True):
			tmpm = Mol(m.atoms,x_)
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(tmpm, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			if DoForce:
				return energy, force
			else:
				return energy
		return EnAndForce

	# This actually affords a 2x speedup on set evaluation.
	def GetEnergyForceForSet(m,batch_size = 200):
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

	F = GetEnergyForceForMol(m)
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

	MOpt = ScannedOptimization(F,m)
	m = MOpt.Search(m)
	exit(0)

#Eval()
#TestBetaHairpin()
#TestUrey()
ConfSearch()
