import TensorMol as TM
import numpy as np

qm_atom_count = 3
full_xyz    = '''6
water dimer
H     0.97211484     2.12293143     0.40228449
H     0.14915974     2.86873778     1.41141964
O     1.03113538     2.86271140     1.02966233
H     0.12111365    -0.04822847     0.12190870
H     1.53628287    -0.00450934    -0.39550118
O     0.69394964     0.45590908    -0.46204465
'''

starting_molecule = TM.Mol()
starting_molecule.FromXYZString(full_xyz)
atoms = starting_molecule.atoms

a=TM.MSet()
a.mols.append(starting_molecule)

def GetChemSpiderNetwork():
	TreatedAtoms = np.array([1,6,7,8], dtype=np.uint8)
	# TM.PARAMS["networks_directory"] = "/home/animal/Packages/TensorMol/networks/"
	TM.PARAMS["tf_prec"] = "tf.float64"
	TM.PARAMS["NeuronType"] = "sigmoid_with_param"
	TM.PARAMS["sigmoid_alpha"] = 100.0
	TM.PARAMS["HiddenLayers"] = [2000, 2000, 2000]
	TM.PARAMS["EECutoff"] = 15.0
	TM.PARAMS["EECutoffOn"] = 0
	TM.PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	TM.PARAMS["EECutoffOff"] = 15.0
	TM.PARAMS["AddEcc"] = True
	TM.PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]

	d = TM.MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
	tset = TM.TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	TM.PARAMS["DSFAlpha"] = 0.18*TM.BOHRPERA

	manager=TM.TFMolManage("chemspider12_solvation", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	return manager

manager = GetChemSpiderNetwork()
print("~~~~~ QM-ML Energy: {} ~~~~~".format(TM.QMMLEnergy(manager, starting_molecule, 3)))
