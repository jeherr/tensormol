from TensorMol import *
from berny import Berny, geomlib

xyz = """28
molecule
C     0.255595    -1.413301     0.025199
C     1.415790    -0.638604     0.008902
C     1.325079     0.752964    -0.014147
C     0.073723     1.370428    -0.022107
C    -1.086151     0.595844    -0.006292
C    -0.995237    -0.796121     0.017759
H     2.401772    -1.125468     0.015550
H    -2.072514     1.082180    -0.012568
O    -0.007837     2.626008    -0.042972
O     0.337692    -2.668840     0.046509
N     2.547452     1.569320    -0.030472
N    -2.217803    -1.612175     0.034686
C    -3.007146    -1.287461     1.231515
H    -3.540069    -0.373747     1.070209
H    -3.703072    -2.076911     1.424793
H    -2.352854    -1.175369     2.070703
C     3.342368     1.285452     1.173034
H     3.861359     0.358247     1.047186
H     4.050637     2.072657     1.326542
H     2.693720     1.218829     2.021396
C     3.337267     1.245094    -1.227121
H     4.092771     1.989526    -1.368311
H     3.798746     0.287954    -1.101342
H     2.695182     1.221801    -2.082741
C    -3.013045    -1.328410    -1.168629
H    -3.778755    -2.068498    -1.272830
H    -3.461692    -0.361087    -1.079728
H    -2.377537    -1.349094    -2.029211"""


def GetChemSpiderNetwork(a, Solvation_=False):
    TreatedAtoms = np.array([1, 6, 7, 8], dtype=np.uint8)
    PARAMS["tf_prec"] = "tf.float64"
    PARAMS["NeuronType"] = "sigmoid_with_param"
    PARAMS["sigmoid_alpha"] = 100.0
    PARAMS["HiddenLayers"] = [2000, 2000, 2000]
    PARAMS["EECutoff"] = 15.0
    PARAMS["EECutoffOn"] = 0
    PARAMS[
        "Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
    PARAMS["EECutoffOff"] = 15.0
    PARAMS["AddEcc"] = True
    PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]
    d = MolDigester(
        TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole"
    )  # Initialize a digester that apply descriptor for the fragme
    tset = TensorMolData_BP_Direct_EE_WithEle(
        a, d, order_=1, num_indis_=1, type_="mol", WithGrad_=True)
    if Solvation_:
        PARAMS["DSFAlpha"] = 0.18
        manager = TFMolManage(
            "chemspider12_solvation", tset, False,
            "fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",
            False, False)
    else:
        PARAMS["DSFAlpha"] = 0.18 * BOHRPERA
        manager = TFMolManage(
            "chemspider12_nosolvation", tset, False,
            "fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",
            False, False)
    return manager


initial_molecule = Mol()
initial_molecule.FromXYZString(xyz)
molecule_set = MSet()
molecule_set.mols.append(initial_molecule)
manager = GetChemSpiderNetwork(molecule_set, False)  # load chemspider network


# Make wrapper functions for energy, force and dipole
def EnergyAndForce(molecule):
    (Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole,
     atom_charge, gradient) = manager.EvalBPDirectEEUpdateSingle(
         molecule, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"],
         PARAMS["EECutoffOff"], True)
    energy = Etotal
    return energy[0], (-1.0 * gradient[0] / JOULEPERHARTREE)


optimizer = Berny(
    geomlib.loads(xyz, 'xyz'),
    maxsteps=1000,
    gradientrms=0.0003,
    gradientmax=0.003,
    debug=True)
debug = []
last_geom = ""

for geom in optimizer:
    molecule_xyz = geom.dumps('xyz')
    molecule = Mol()
    molecule.FromXYZString(molecule_xyz)
    energy, gradients = EnergyAndForce(
        molecule)  # calculate energy and gradients of geom
    info = optimizer.send((energy, gradients))
    debug.append(info)
print('Final Geometry:')
print(geom.dumps('xyz'))
