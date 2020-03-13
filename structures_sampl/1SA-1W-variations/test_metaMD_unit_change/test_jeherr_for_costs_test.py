import sys 
sys.path.append("../../..")
from TensorMol import *
import time
import random
PARAMS["max_checkpoints"] = 3
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Takes two nearly identical crystal lattices and interpolates a core/shell structure, must be oriented identically and stoichiometric





def TestMetadynamics():
	a = MSet("1SA1Winit")
	a.ReadXYZ()
	m = a.mols[-1]
	ForceField = lambda x: QchemDFT(Mol(m.atoms,x),basis_ = '6-311G**',xc_='wB97XD', jobtype_='force', filename_='jmols2', path_='./qchem/', threads=8)
#	manager = TFMolManageDirect(name="BehlerParinelloDirectSymFunc_nicotine_vib_Tue_Nov_21_09.11.26_2017", network_type = "BehlerParinelloDirectSymFunc")
#	def force_field(coords):
#		energy, forces = manager.evaluate_mol(Mol(m.atoms, coords), True)
#		return energy, forces * JOULEPERHARTREE
	masses = np.array(list(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms)))
	print "Masses:", masses
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 10000
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 600.0
	PARAMS["MDV0"] = "Thermal"
	PARAMS["MetaBumpTime"] = 10.0
	PARAMS["MetaMaxBumps"] = 50000
	PARAMS["MetaMDBumpHeight"] = 1.0
	PARAMS["MetaMDBumpWidth"] = 2.0
	meta = MetaDynamics(ForceField, m, EandF_=ForceField)
	meta.Prop()

def test_md():
	PARAMS["RBFS"] = np.array([[0.35, 0.35], [0.70, 0.35], [1.05, 0.35], [1.40, 0.35], [1.75, 0.35], [2.10, 0.35], [2.45, 0.35],
								[2.80, 0.35], [3.15, 0.35], [3.50, 0.35], [3.85, 0.35], [4.20, 0.35], [4.55, 0.35], [4.90, 0.35]])
	PARAMS["ANES"] = np.array([2.20, 1.0, 1.0, 1.0, 1.0, 2.55, 3.04, 3.44]) #pauling electronegativity
	PARAMS["SH_NRAD"] = 14
	PARAMS["SH_LMAX"] = 4
	a = MSet("OptMols")
	a.ReadXYZ()
	mol = a.mols[4]
	manager=TFManage(Name_="SmallMols_GauSH_fc_sqdiff_GauSH_direct",Train_=False,NetType_="fc_sqdiff_GauSH_direct")
	force_field = lambda x: manager.evaluate_mol_forces_direct(x)
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1], mol.atoms))
	print "Masses:", masses
	PARAMS["MDdt"] = 0.2
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 20000
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDTemp"]= 300.0
	md = VelocityVerlet(force_field, mol)
	md.Prop()


def water_dimer_plot():
	PARAMS["RBFS"] = np.stack((np.linspace(0.1, 5.0, 32), np.repeat(0.25, 32)), axis=1)
	PARAMS["SH_NRAD"] = 32
	PARAMS["SH_LMAX"] = 4
	def qchemdft(m_,ghostatoms,basis_ = '6-31g*',xc_='b3lyp', jobtype_='force', filename_='tmp', path_='./qchem/', threads=False):
		istring = '$molecule\n0 1 \n'
		crds = m_.coords.copy()
		crds[abs(crds)<0.0000] *=0.0
		for j in range(len(m_.atoms)):
			if j in ghostatoms:
				istring=istring+"@"+itoa[m_.atoms[j]]+' '+str(crds[j,0])+' '+str(crds[j,1])+' '+str(crds[j,2])+'\n'
			else:
				istring=istring+itoa[m_.atoms[j]]+' '+str(crds[j,0])+' '+str(crds[j,1])+' '+str(crds[j,2])+'\n'
		if jobtype_ == "dipole":
			istring =istring + '$end\n\n$rem\njobtype sp\nbasis '+basis_+'\nmethod '+xc_+'\nthresh 11\nsymmetry false\nsym_ignore true\n$end\n'
		else:
			istring =istring + '$end\n\n$rem\njobtype '+jobtype_+'\nbasis '+basis_+'\nmethod '+xc_+'\nthresh 11\nsymmetry false\nsym_ignore true\n$end\n'
		with open(path_+filename_+'.in','w') as fin:
			fin.write(istring)
		with open(path_+filename_+'.out','a') as fout:
			if threads:
				proc = subprocess.Popen(['qchem', '-nt', str(threads), path_+filename_+'.in'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
			else:
				proc = subprocess.Popen(['qchem', path_+filename_+'.in'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
			out, err = proc.communicate()
			fout.write(out)
		lines = out.split('\n')
		if jobtype_ == 'force':
			Forces = np.zeros((m_.atoms.shape[0],3))
			for i, line in enumerate(lines):
				if line.count('Convergence criterion met')>0:
					Energy = float(line.split()[1])
				if line.count("Gradient of SCF Energy") > 0:
					k = 0
					l = 0
					for j in range(1, m_.atoms.shape[0]+1):
						Forces[j-1,:] = float(lines[i+k+2].split()[l+1]), float(lines[i+k+3].split()[l+1]), float(lines[i+k+4].split()[l+1])
						l += 1
						if (j % 6) == 0:
							k += 4
							l = 0
			# return Energy, Forces
			return Energy, -Forces*JOULEPERHARTREE/BOHRPERA
		elif jobtype_ == 'sp':
			for line in lines:
				if line.count('Convergence criterion met')>0:
					Energy = float(line.split()[1])
			return Energy
		else:
			raise Exception("jobtype needs formatted for return variables")

	a = MSet("water_dimer")
	a.ReadXYZ()
	manager = TFMolManageDirect(name="BehlerParinelloDirectGauSH_H2O_wb97xd_1to21_with_prontonated_Mon_Nov_13_11.35.07_2017", network_type = "BehlerParinelloDirectGauSH")
	qchemff = lambda x, y: qchemdft(x, y, basis_ = '6-311g**',xc_='wb97x-d', jobtype_='sp', filename_='tmp', path_='./qchem/', threads=8)
	cp_correction = []
	for mol in a.mols:
		h2o1 = qchemff(Mol(mol.atoms[:3], mol.coords[:3]), [])
		h2o2 = qchemff(Mol(mol.atoms[3:], mol.coords[3:]), [])
		# h2o1cp = qchemff(mol, [3, 4, 5])
		# h2o2cp = qchemff(mol, [0, 1, 2])
		dimer = qchemff(mol, [])
		# cpc = h2o1cp - h2o1 + h2o2cp - h2o2
		# cp_correction.append(cpc)
		bond_e = dimer - h2o1 - h2o2
		print "{%.10f, %.10f}," % (np.linalg.norm(mol.coords[1] - mol.coords[3]), bond_e * 627.509)
	print "TensorMol evaluation"
	for i, mol in enumerate(a.mols):
		h2o1 = manager.evaluate_mol(Mol(mol.atoms[:3], mol.coords[:3]), False)
		h2o2 = manager.evaluate_mol(Mol(mol.atoms[3:], mol.coords[3:]), False)
		dimer = manager.evaluate_mol(mol, False)
		bond_e = dimer - h2o1 - h2o2
		print "{%.10f, %.10f}," % (np.linalg.norm(mol.coords[1] - mol.coords[3]), bond_e * 627.509)


def GetWaterNetwork():
	a=MSet("1H2Oinit")
	a.ReadXYZ()
	TreatedAtoms = a.AtomTypes()
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 50000
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 300.0
	PARAMS["MDV0"] = "Random"
	PARAMS["MetaMDBumpHeight"] = 1.0
	PARAMS["MetaMDBumpWidth"] = 2.0
	PARAMS["MetaBowlK"] = 0.2
	PARAMS["MetaBumpTime"] = 5.0
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["NeuronType"] = "sigmoid_with_param"
	PARAMS["sigmoid_alpha"] = 100.0
	PARAMS["HiddenLayers"] = [500, 500, 500]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	PARAMS["EECutoffOff"] = 15.0
	PARAMS["DSFAlpha"] = 0.18
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 1.0]
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("water_network",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	def EnAndForce(x_, DoForce=True):
		mtmp = Mol(m.atoms,x_)
		Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
		energy = Etotal[0]
		force = gradient[0]
		if DoForce:
			return energy, force
		else:
			return energy
	m=a.mols[0]
	PARAMS["OptMaxCycles"]= 2000
	PARAMS["OptThresh"] =0.00002
	Opt = GeomOptimizer(EnAndForce)
	mo=Opt.Opt(a.mols[0],"morphine_tm_opt")
	mo.WriteXYZfile("./results/", "opt_h2o_hex_bag")
	masses = np.array(list(map(lambda x: ATOMICMASSESAMU[x-1],mo.atoms)))
	meta = MetaDynamics(EnAndForce, mo, EandF_=EnAndForce, name_="water_hexamer")
	meta.Prop()

def water_meta_opt():
	a=MSet("water10")
	a.ReadXYZ()
	TreatedAtoms = a.AtomTypes()
	m=a.mols[0]
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"] = True
	PARAMS["MDMaxStep"] = 50000
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 600.0
	PARAMS["MDV0"] = "Random"
	PARAMS["MetaMDBumpHeight"] = 1.0
	PARAMS["MetaMDBumpWidth"] = 2.0
	PARAMS["MetaBowlK"] = 0.2
	PARAMS["MetaBumpTime"] = 5.0
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["NeuronType"] = "sigmoid_with_param"
	PARAMS["sigmoid_alpha"] = 100.0
	PARAMS["HiddenLayers"] = [500, 500, 500]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	PARAMS["EECutoffOff"] = 15.0
	PARAMS["DSFAlpha"] = 0.18
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 1.0]
	PARAMS["OptMaxCycles"]= 2000
	PARAMS["OptThresh"] =0.00002
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("water_network",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	atomization_energy = 0.0
	for atom in mol.atoms:
		if atom in ele_U:
			atomization_energy += ele_U[atom]
	def EnAndForce(x_, DoForce=True):
		mtmp = Mol(m.atoms,x_)
		Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
		energy = Etotal[0] + atomization_energy
		force = gradient[0]
		if DoForce:
			return energy, force
		else:
			return energy

def water_meta_react():
	a=MSet("water10")
	a.ReadXYZ()
	TreatedAtoms = a.AtomTypes()
	m=a.mols[0]
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"] = True
	PARAMS["MDMaxStep"] = 10000
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 600.0
	PARAMS["MDV0"] = "Random"
	PARAMS["MetaMDBumpHeight"] = 2.0
	PARAMS["MetaMDBumpWidth"] = 3.0
	PARAMS["MetaBowlK"] = 0.2
	PARAMS["MetaBumpTime"] = 5.0
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["NeuronType"] = "sigmoid_with_param"
	PARAMS["sigmoid_alpha"] = 100.0
	PARAMS["HiddenLayers"] = [500, 500, 500]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	PARAMS["EECutoffOff"] = 15.0
	PARAMS["DSFAlpha"] = 0.18
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 1.0]
	PARAMS["OptMaxCycles"]= 2000
	PARAMS["OptThresh"] =0.00002
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("water_network",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	atomization_energy = 0.0
	for atom in m.atoms:
		if atom in ele_U:
			atomization_energy += ele_U[atom]
	def EnAndForce(x_, DoForce=True):
		mtmp = Mol(m.atoms,x_)
		Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
		energy = Etotal[0] + atomization_energy
		force = gradient[0]
		if DoForce:
			return energy, force
		else:
			return energy
	meta = MetaDynamics(EnAndForce, m,name_="water_10react", EandF_=EnAndForce)
	meta.Prop()



# PARAMS["RBFS"] = np.stack((np.linspace(0.1, 5.0, 32), np.repeat(0.25, 32)), axis=1)
# PARAMS["SH_NRAD"] = 32
# PARAMS["SH_LMAX"] = 4
# a = MSet("water_dimer")
# a.ReadXYZ()
# manager = TFMolManageDirect(name="BehlerParinelloDirectGauSH_H2O_wb97xd_1to21_with_prontonated_Mon_Nov_13_11.35.07_2017", network_type = "BehlerParinelloDirectGauSH")
# print manager.evaluate_mol(a.mols[0], False).shape
# for i in range(100):
# 	mol = Mol(a.mols[0].atoms, rot_coords[0,i])
#  	mol.WriteXYZfile()

# InterpoleGeometries()
# read_unpacked_set()
# TrainKRR(set_="SmallMols_rand", dig_ = "GauSH", OType_="Force")
# RandomSmallSet("SmallMols", 10000)
TestMetadynamics()

# test_md()
# TestTFBond()
# TestTFGauSH()
# test_gaussian_overlap()
# train_forces_GauSH_direct("SmallMols_rand")
# test_tf_neighbor()
# train_energy_pairs_triples()
# train_energy_symm_func("water_wb97xd_6311gss")
# train_energy_GauSH("water_wb97xd_6311gss")
# test_h2o()
# evaluate_BPSymFunc("nicotine_vib")
# water_dimer_plot()
# nicotine_cc_stretch_plot()
# meta_statistics()
# meta_stat_plot()
# harmonic_freq()
# train_Poly_GauSH()
#water_ir()
#GetWaterNetwork()
# water_meta_opt()
#water_meta_react()
# meta_opt()
# water_web()

# f=open("nicotine_md_aimd_log.dat","r")
# f2=open("nicotine_md_aimd_energies.dat", "w")
# lines=f.readlines()
# for line in lines:
# 	f2.write(str(float(line.split()[0])/1000.0)+" "+str(float(line.split()[7]) * 627.509)+"\n")
# f.close()
# f2.close()

# import pickle
# water_data = pickle.load(open("./datasets/H2O_wbxd_1to21_with_prontonated.dat","rb"))
# a=MSet("water_clusters")
# for i, mol in enumerate(water_data):
# 	a.mols.append(Mol(np.array(mol["atoms"]), mol["xyz"]))
# 	a.mols[-1].properties["name"] = mol["name"]
# 	a.mols[-1].properties["energy"] = mol["scf_energy"]
# 	a.mols[-1].properties["dipole"] = np.array(mol["dipole"])
# 	a.mols[-1].properties["gradients"] = mol["gradients"]
# 	try:
# 		a.mols[-1].properties["quadrupole"] = np.array(mol["quad"])
# 		a.mols[-1].properties["mulliken_charges"] = np.array(mol["charges"])
# 	except Exception as Ex:
# 		print Ex
# 		print i
# 		pass
# a.Save()

# PARAMS["tf_prec"] = "tf.float32"
# PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 16), np.repeat(0.35, 16)), axis=1)
# PARAMS["SH_NRAD"] = 16
# a = MSet("SmallMols_rand")
# a.Load()
# # a.mols.append(Mol(np.array([1,1,8]),np.array([[0.9,0.1,0.1],[1.,0.9,1.],[0.1,0.1,0.1]])))
# # # # Tesselate that water to create a box
# # ntess = 16
# # latv = 2.8*np.eye(3)
# # # # # Start with a water in a ten angstrom box.
# # lat = Lattice(latv)
# # mc = lat.CenteredInLattice(a.mols[0])
# # mt = Mol(*lat.TessNTimes(mc.atoms,mc.coords,ntess))
# # # # mt.WriteXYZfile()
# b=MSet()
# for i in range(1):
# 	b.mols.append(a.mols[i])
# 	new_mol = copy.deepcopy(a.mols[i])
# 	new_mol.RotateRandomUniform()
# 	b.mols.append(new_mol)
# # # a=MSet("SmallMols_rand")
# # # a.Load()
# maxnatoms = b.MaxNAtoms()
# zlist = []
# xyzlist = []
# n_atoms_list = []
# for i, mol in enumerate(b.mols):
# 	paddedxyz = np.zeros((maxnatoms,3), dtype=np.float32)
# 	paddedxyz[:mol.atoms.shape[0]] = mol.coords
# 	paddedz = np.zeros((maxnatoms), dtype=np.int32)
# 	paddedz[:mol.atoms.shape[0]] = mol.atoms
# 	xyzlist.append(paddedxyz)
# 	zlist.append(paddedz)
# 	n_atoms_list.append(mol.NAtoms())
# 	if i == 99:
# 		break
# xyzstack = tf.stack(xyzlist)
# zstack = tf.stack(zlist)
# natomsstack = tf.stack(n_atoms_list)
# r_cutoff = 7.0
# gaussian_params = tf.Variable(PARAMS["RBFS"], trainable=True, dtype=tf.float32)
# # atomic_embed_factors = tf.Variable(PARAMS["ANES"], trainable=True, dtype=tf.float32)
# elements = tf.constant([1, 8], dtype=tf.int32)
# # tmp = tf_neighbor_list_sort(xyzstack, zstack, natomsstack, elements, r_cutoff)
# # tmp = tf_sparse_gauss_harmonics_echannel(xyzstack, zstack, natomsstack, elements, gaussian_params, 4, r_cutoff)
# tmp2 = tf_gauss_harmonics_echannel(xyzstack, zstack, elements, gaussian_params, 8)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# # run_metadata = tf.RunMetadata()
# # # for i in range(a.mols[0].atoms.shape[0]):
# # # 	print a.mols[0].atoms[i], "   ", a.mols[0].coords[i,0], "   ", a.mols[0].coords[i,1], "   ", a.mols[0].coords[i,2]
# @TMTiming("test")
# def get_pairs():
# 	tmp3 = sess.run(tmp2)
# 	return tmp3
# tmp5 = get_pairs()
# print tmp5[:13].shape
# print tmp5[13:].shape
# print np.allclose(tmp5[:13], tmp5[13:], 1e-03)
# # print np.isclose(tmp5[0][0], tmp6[0][0], 1e-01)
# # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
# # chrome_trace = fetched_timeline.generate_chrome_trace_format()
# # with open('timeline_step_tmp_tm_nocheck_h2o.json', 'w') as f:
# # 	f.write(chrome_trace)
