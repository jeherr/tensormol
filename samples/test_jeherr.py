from TensorMol import *
import time
import random
import itertools as it
PARAMS["max_checkpoints"] = 3
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Takes two nearly identical crystal lattices and interpolates a core/shell structure, must be oriented identically and stoichiometric
def InterpolateGeometries():
	a=MSet('cspbbr3_tess')
	#a.ReadGDB9Unpacked(path='/media/sdb2/jeherr/TensorMol/datasets/cspbbr3/pb_tess_6sc/')
	#a.Save()
	a.Load()
	mol1 = a.mols[0]
	mol2 = a.mols[1]
	mol2.RotateX()
	mol1.AlignAtoms(mol2)
	optimizer = Optimizer(None)
	optimizer.Interpolate_OptForce(mol1, mol2)
	mol1.WriteXYZfile(fpath='./results/cspbbr3_tess', fname='cspbbr3_6sc_pb_tess_goopt', mode='w')
	# mol2.WriteXYZfile(fpath='./results/cspbbr3_tess', fname='cspbbr3_6sc_ortho_rot', mode='w')

def read_unpacked_set(set_name="chemspider12", paths="/media/sdb2/jeherr/TensorMol/datasets/chemspider12/*/", properties=["name", "energy", "gradients", "dipole"]):
	import glob
	a=MSet(set_name)
	for path in glob.iglob(paths):
		a.read_xyz_set_with_properties(paths, properties)
	print(len(a.mols), " Molecules")
	a.Save()

def TrainKRR(set_ = "SmallMols", dig_ = "GauSH", OType_ ="Force"):
	a=MSet("SmallMols_rand")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms, name_=dig_,OType_ =OType_)
	tset = TensorData(a,d)
	tset.BuildTrainMolwise("SmallMols",TreatedAtoms)
	manager=TFManage("",tset,True,"KRR_sqdiff")
	return

def RandomSmallSet(set_, size_):
	""" Returns an MSet of random molecules chosen from a larger set """
	print("Selecting a subset of "+str(set_)+" of size "+str(size_))
	a=MSet(set_)
	a.Load()
	b=MSet(set_+"_rand")
	mols = random.sample(range(len(a.mols)), size_)
	for i in mols:
		b.mols.append(a.mols[i])
	b.Save()
	return b

def TestMetadynamics():
	a = MSet("nicotine_opt")
	a.ReadXYZ()
	m = a.mols[-1]
	# ForceField = lambda x: QchemDFT(Mol(m.atoms,x),basis_ = '6-311g**',xc_='wB97X-D', jobtype_='force', filename_='jmols2', path_='./qchem/', threads=8)
	manager = TFMolManageDirect(name="BehlerParinelloDirectSymFunc_nicotine_vib_Tue_Nov_21_09.11.26_2017", network_type = "BehlerParinelloDirectSymFunc")
	def force_field(coords):
		energy, forces = manager.evaluate_mol(Mol(m.atoms, coords), True)
		return energy, forces * JOULEPERHARTREE
	masses = np.array(list(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms)))
	print("Masses:", masses)
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 50000
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 300.0
	PARAMS["MDV0"] = "Thermal"
	PARAMS["MetaMDBumpHeight"] = 0.00
	PARAMS["MetaMDBumpWidth"] = 0.01
	meta = MetaDynamics(force_field, m, EandF_=force_field)
	meta.Prop()

def test_md():
	PARAMS["tf_prec"] = "tf.float32"
	a = MSet("water10")
	a.ReadXYZ()
	mol = a.mols[1]
	network = BehlerParinelloGauSHv2(name="BPGauSH_water_wb97xd_6311gss_Sat_Apr_14_00.23.37_2018")
	def force_field(coords, forces=True):
		m=Mol(mol.atoms, coords)
		if forces:
			energy, forces = network.evaluate_mol(m, forces)
			return energy, JOULEPERHARTREE*forces
		else:
			energy = network.evaluate_mol(m, forces)
			return energy
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1], mol.atoms))
	print("Masses:", masses)
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 20000
	PARAMS["MDThermostat"] = None
	PARAMS["MDTemp"]= 300.0
	md = VelocityVerlet(force_field, mol, EandF_=force_field)
	md.Prop()

def TestTFBond():
	a=MSet("chemspider_all_rand")
	a.Load()
	d = MolDigester(a.BondTypes(), name_="CZ", OType_="AtomizationEnergy")
	tset = TensorMolData_BPBond_Direct(a,d)
	manager=TFMolManage("",tset,True,"fc_sqdiff_BPBond_Direct")

def TestTFGauSH():
	tf_precision = eval(PARAMS["tf_prec"])
	TensorMol.RawEmbeddings.data_precision = tf_precision
	np.set_printoptions(threshold=100000)
	a=MSet("SmallMols_rand")
	a.Load()
	MaxNAtom = a.MaxNAtom()
	zlist = []
	xyzlist = []
	labelslist = []
	natomlist = []
	for i, mol in enumerate(a.mols):
		paddedxyz = np.zeros((MaxNAtom,3), dtype=np.float32)
		paddedxyz[:mol.atoms.shape[0]] = mol.coords
		paddedz = np.zeros((MaxNAtom), dtype=np.int32)
		paddedz[:mol.atoms.shape[0]] = mol.atoms
		paddedlabels = np.zeros((MaxNAtom, 3), dtype=np.float32)
		paddedlabels[:mol.atoms.shape[0]] = mol.properties["forces"]
		xyzlist.append(paddedxyz)
		zlist.append(paddedz)
		labelslist.append(paddedlabels)
		natomlist.append(mol.NAtoms())
		if i == 999:
			break
	xyzstack = tf.stack(xyzlist)
	zstack = tf.stack(zlist)
	labelstack = tf.stack(labelslist)
	natomstack = tf.stack(natomlist)
	gaussian_params = tf.Variable(PARAMS["RBFS"], trainable=True, dtype=tf.float32)
	atomic_embed_factors = tf.Variable(PARAMS["ANES"], trainable=True, dtype=tf.float32)
	elements = tf.constant([1, 6, 7, 8], dtype=tf.int32)
	tmp = tf_gaussian_spherical_harmonics_channel(xyzstack, zstack, elements, gaussian_params, 4)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	# for i in range(a.mols[0].atoms.shape[0]):
	# 	print(a.mols[0].atoms[i], "   ", a.mols[0].coords[i,0], "   ", a.mols[0].coords[i,1], "   ", a.mols[0].coords[i,2])
	tmp2 = sess.run(tmp, options=options, run_metadata=run_metadata)
	print(tmp2)
	# print(tmp2[1])
	# print(tmp2.shape)
	# print(tmp3)
	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
	with open('timeline_step_tmp_tm_nocheck_h2o.json', 'w') as f:
		f.write(chrome_trace)
	# print(tmp2[3].shape)
	# print(a.mols[0].atoms.shape)
	# TreatedAtoms = a.AtomTypes()
	# d = Digester(TreatedAtoms, name_="GauSH", OType_="Force")
	# # tset = TensorData(a,d)
	# mol_ = a.mols[0]
	# print(d.Emb(mol_, -1, mol_.coords[0], MakeOutputs=False)[0])
	# print(mol_.atoms[0])

def test_gaussian_overlap():
	gaussian_params = tf.Variable(PARAMS["RBFS"], trainable=True, dtype=tf.float32)
	tf_precision = eval(PARAMS["tf_prec"])
	TensorMol.RawEmbeddings.data_precision = tf_precision
	tmp = tf_gaussian_overlap(gaussian_params)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	tmp2 = sess.run(tmp)
	print(tmp2)

def train_forces_GauSH_direct(set_ = "SmallMols"):
	PARAMS["RBFS"] = np.array([[0.35, 0.35], [0.70, 0.35], [1.05, 0.35], [1.40, 0.35], [1.75, 0.35], [2.10, 0.35], [2.45, 0.35],[2.80, 0.35], [3.15, 0.35], [3.50, 0.35], [3.85, 0.35], [4.20, 0.35], [4.55, 0.35], [4.90, 0.35]])
	PARAMS["ANES"] = np.array([2.20, 1.0, 1.0, 1.0, 1.0, 2.55, 3.04, 3.44]) #pauling electronegativity
	PARAMS["SH_NRAD"] = 14
	PARAMS["SH_LMAX"] = 4
	PARAMS["HiddenLayers"] = [512, 512, 512, 512, 512, 512, 512]
	PARAMS["max_steps"] = 20000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 200
	PARAMS["NeuronType"] = "elu"
	PARAMS["learning_rate"] = 0.0001
	a=MSet(set_)
	a.Load()
	TreatedAtoms = a.AtomTypes()
	print("Number of Mols: ", len(a.mols))
	d = Digester(TreatedAtoms, name_="GauSH", OType_="Force")
	tset = TensorDataDirect(a,d)
	manager=TFManage("",tset,True,"fc_sqdiff_GauSH_direct")

def test_tf_neighbor():
	np.set_printoptions(threshold=100000)
	a=MSet("SmallMols_rand")
	a.Load()
	MaxNAtom = a.MaxNAtom()
	zlist = []
	xyzlist = []
	labelslist = []
	for i, mol in enumerate(a.mols):
		paddedxyz = np.zeros((MaxNAtom,3), dtype=np.float32)
		paddedxyz[:mol.atoms.shape[0]] = mol.coords
		paddedz = np.zeros((MaxNAtom), dtype=np.int32)
		paddedz[:mol.atoms.shape[0]] = mol.atoms
		paddedlabels = np.zeros((MaxNAtom, 3), dtype=np.float32)
		paddedlabels[:mol.atoms.shape[0]] = mol.properties["forces"]
		xyzlist.append(paddedxyz)
		zlist.append(paddedz)
		labelslist.append(paddedlabels)
		if i == 99:
			break
	xyzstack = tf.stack(xyzlist)
	zstack = tf.stack(zlist)
	labelstack = tf.stack(labelslist)
	gaussian_params = tf.Variable(PARAMS["RBFS"], trainable=True, dtype=tf.float32)
	atomic_embed_factors = tf.Variable(PARAMS["ANES"], trainable=True, dtype=tf.float32)
	element = tf.constant(1, dtype=tf.int32)
	r_cutoff = tf.constant(5.0, dtype=tf.float32)
	element_pairs = tf.constant([[1,1,1], [1,1,6], [1,1,7], [1,1,8], [1,6,6], [1,6,7], [1,6,8], [1,7,7], [1,7,8], [1,8,8],
								[6,6,6], [6,6,7], [6,6,8], [6,7,7], [6,7,8], [6,8,8], [7,7,7], [7,7,8], [7,8,8], [8,8,8]], dtype=tf.int32)
	tmp = tf_triples_list(xyzstack, zstack, r_cutoff, element_pairs)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	# for i in range(a.mols[0].atoms.shape[0]):
	# 	print(a.mols[0].atoms[i], "   ", a.mols[0].coords[i,0], "   ", a.mols[0].coords[i,1], "   ", a.mols[0].coords[i,2])
	tmp3 = sess.run([tmp], options=options, run_metadata=run_metadata)
	# print(tmp3)
	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
	with open('timeline_step_tmp_tm_nocheck_h2o.json', 'w') as f:
		f.write(chrome_trace)
	print(tmp3)
	# print(tmp4[1])
	# print(tmp4)
	# TreatedAtoms = a.AtomTypes()
	# d = Digester(TreatedAtoms, name_="GauSH", OType_="Force")
	# # tset = TensorData(a,d)
	# mol_ = a.mols[0]
	# print(d.Emb(mol_, -1, mol_.coords[0], MakeOutputs=False)[0])
	# print(mol_.atoms[0])

def train_energy_pairs_triples():
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 200
	PARAMS["NeuronType"] = "relu"
	# PARAMS["tf_prec"] = "tf.float64"
	# PARAMS["self.profiling"] = True
	a=MSet("SmallMols")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	print("Number of Mols: ", len(a.mols))
	d = Digester(TreatedAtoms, name_="GauSH", OType_="AtomizationEnergy")
	tset = TensorMolData_BP_Direct(a,d)
	manager=TFMolManage("",tset,True,"pairs_triples", Trainable_=True)

def train_energy_symm_func(mset):
	PARAMS["train_energy_gradients"] = False
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 200
	PARAMS["NeuronType"] = "elu"
	PARAMS["tf_prec"] = "tf.float64"
	a=MSet(mset)
	a.Load()
	print("Number of Mols: ", len(a.mols))
	manager = TFMolManageDirect(a, network_type = "BPSymFunc")

def train_energy_GauSH(mset):
	PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 16), np.repeat(0.30, 16)), axis=1)
	PARAMS["SH_NRAD"] = 16
	PARAMS["SH_LMAX"] = 5
	PARAMS["SH_rot_invar"] = False
	PARAMS["EECutoffOn"] = 0.0
	PARAMS["Elu_Width"] = 6.0
	PARAMS["train_gradients"] = False
	PARAMS["train_dipole"] = False
	PARAMS["train_rotation"] = False
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.00005
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 100
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float32"
	PARAMS["Profiling"] = False
	PARAMS["train_sparse"] = False
	PARAMS["sparse_cutoff"] = 7.0
	network = BehlerParinelloGauSH(mset)
	network.start_training()

def train_energy_GauSHv2(mset):
	PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 16), np.repeat(0.30, 16)), axis=1)
	PARAMS["SH_LMAX"] = 5
	PARAMS["train_gradients"] = True
	PARAMS["train_dipole"] = False
	PARAMS["train_rotation"] = False
	PARAMS["train_sparse"] = False
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 100
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float32"
	network = BehlerParinelloGauSHv2(mset)
	network.start_training()

def train_energy_univ(mset):
	PARAMS["train_gradients"] = True
	PARAMS["train_charges"] = True
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [1024, 1024, 1024]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 32
	PARAMS["Profiling"] = False
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float32"
	network = UniversalNetwork_v2(mset)
	network.start_training()

def eval_test_set_univ(mset):
	PARAMS["train_gradients"] = True
	PARAMS["train_charges"] = True
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [1024, 1024, 1024]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 100
	PARAMS["Profiling"] = False
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float64"
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Tue_May_15_10.18.25_2018")
	molset = MSet(mset)
	molset.Load()
	energy_errors, gradient_errors, charge_errors = network.evaluate_set(molset)
	mae_e = np.mean(np.abs(energy_errors))
	mse_e = np.mean(energy_errors)
	rmse_e = np.sqrt(np.mean(np.square(energy_errors)))
	mae_g = np.mean(np.abs(gradient_errors))
	mse_g = np.mean(gradient_errors)
	rmse_g = np.sqrt(np.mean(np.square(gradient_errors)))
	mae_c = np.mean(np.abs(charge_errors))
	mse_c = np.mean(charge_errors)
	rmse_c = np.sqrt(np.mean(np.square(charge_errors)))
	print("MAE  Energy: ", mae_e, " Gradients: ", mae_g, " Charges: ", mae_c)
	print("MSE  Energy: ", mse_e, " Gradients: ", mse_g, " Charges: ", mse_c)
	print("RMSE  Energy: ", rmse_e, " Gradients: ", rmse_g, " Charges: ", rmse_c)

def test_h2o():
	PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 16), np.repeat(0.30, 16)), axis=1)
	PARAMS["SH_NRAD"] = 16
	PARAMS["SH_LMAX"] = 3
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["OptMaxCycles"]=500
	PARAMS["OptStepSize"] = 0.1
	PARAMS["OptThresh"]=0.0001
	PARAMS["MDAnnealT0"] = 20.0
	PARAMS["MDAnnealSteps"] = 2000
	a = MSet("water_tmp")
	a.ReadXYZ()
	a.mols.append(Mol(np.array([1,1,8]),np.array([[0.9,0.1,0.1],[1.,0.9,1.],[0.1,0.1,0.1]])))
	m = a.mols[0]
	manager = TFMolManageDirect(name="BPGauSH_water_wb97xd_6311gss_Mon_Feb_12_12.17.58_2018", network_type = "BPGauSH")
	def force_field(coords, eval_forces=True):
		mol = Mol()
		mol.atoms = m.atoms
		mol.coords = coords
		if eval_forces:
			energy, forces = manager.evaluate_mol(mol, True)
			forces = RemoveInvariantForce(mol.coords, forces, mol.atoms)
			return energy*JOULEPERHARTREE, forces*JOULEPERHARTREE
		else:
			energy = manager.evaluate_mol(mol, False)
			return energy*JOULEPERHARTREE
	Opt = GeomOptimizer(force_field)
	opt_mol = Opt.Opt(m)

def evaluate_BPSymFunc(mset):
	a=MSet(mset)
	a.Load()
	output, labels = [], []
	manager = TFMolManageDirect(name="BehlerParinelloDirectSymFunc_nicotine_metamd_10000_Tue_Nov_07_22.35.07_2017", network_type = "BehlerParinelloDirectSymFunc")
	random.shuffle(a.mols)
	batch = []
	for i in range(len(a.mols) / 100):
		for j in range(100):
			labels.append(a.mols[i*100+j].properties["atomization"])
			batch.append(a.mols[i*100+j])
		output.append(manager.evaluate_batch(batch, eval_forces=False))
		batch = []
	output = np.concatenate(output)
	labels = np.array(labels)
	print("MAE:", np.mean(np.abs(output-labels))*627.509)
	print("RMSE:",np.sqrt(np.mean(np.square(output-labels)))*627.509)

def water_dimer_plot():
	a = MSet("water_trimer_stretch")
	a.ReadXYZ()
	m=a.mols[0]
	# PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 16), np.repeat(0.30, 16)), axis=1)
	# PARAMS["SH_NRAD"] = 16
	# PARAMS["SH_LMAX"] = 5
	# PARAMS["HiddenLayers"] = [512, 512, 512]
	# PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float32"
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
	# a = MSet("H2O_dimer_flip_rightone")
	# a.ReadXYZ()
	# manager=TFMolManageDirect(name="BPGauSH_water_wb97xd_6311gss_Mon_Feb_12_12.17.58_2018", network_type = "BPGauSH")
	# manager.network.embed_shape = manager.network.embedding_shape
	network = BehlerParinelloGauSHv2(name="BPGauSH_water_wb97xd_6311gss_Thu_Mar_15_16.29.21_2018")
	qchemff = lambda x, y: qchemdft(x, y, basis_ = '6-311g**',xc_='wb97x-d', jobtype_='sp', filename_='tmp', path_='./qchem/', threads=8)
	# for i in range(len(a.mols)):
	# # 	h2o1 = qchemff(Mol(a.mols[i].atoms[:3], a.mols[i].coords[:3]), [])
	# # 	h2o2 = qchemff(Mol(a.mols[i].atoms[3:], a.mols[i].coords[3:]), [])
	# # # 	# h2o1cp = qchemff(mol, [3, 4, 5])
	# # # 	# h2o2cp = qchemff(mol, [0, 1, 2])
	# 	dimer = qchemff(a.mols[i], [])
	# # # 	# cpc = h2o1cp - h2o1 + h2o2cp - h2o2
	# # # 	# cp_correction.append(cpc)
	# 	bond_e = dimer# - h2o1 - h2o2
	# 	print("{%.10f, %.10f}," % (i, bond_e * 627.509))
	# print("TensorMol evaluation")
	for i in range(len(a.mols)):
		dimer = network.evaluate_mol(a.mols[i], False)
		h2o1 = network.evaluate_mol(Mol(a.mols[i].atoms[:3], a.mols[i].coords[:3]), False)
		h2o2 = network.evaluate_mol(Mol(a.mols[i].atoms[3:], a.mols[i].coords[3:]), False)
		bond_e = dimer - h2o1 - h2o2
		bond_dist = np.linalg.norm(a.mols[i].coords[1] - (a.mols[i].coords[4] + a.mols[i].coords[7])/2)
		print("{%.10f, %.10f}," % (bond_dist, bond_e * 627.509))

def train_Poly_GauSH():
	PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 16), np.repeat(0.35, 16)), axis=1)
	PARAMS["SH_NRAD"] = 16
	PARAMS["SH_LMAX"] = 4
	PARAMS["EECutoffOn"] = 0.0
	PARAMS["Elu_Width"] = 6.0
	PARAMS["train_gradients"] = False
	PARAMS["train_dipole"] = False
	PARAMS["train_rotation"] = True
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 500
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 400
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float32"
	PARAMS["Profiling"] = False
	a=MSet("H2O_augmented_more_cutoff5_b3lyp_force")
	a.Load()
	manager = TFMolManageDirect(a, network_type = "BehlerParinelloDirectGauSH")

def GetWaterNetwork():
	a=MSet("water_hexamer_bag")
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
	PARAMS["MDMaxStep"] = 50000
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

def meta_opt():
	a=MSet("1,5-hexadiene")
	a.ReadXYZ()
	TreatedAtoms = a.AtomTypes()
	m=a.mols[0]
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"] = True
	PARAMS["MDMaxStep"] = 50000
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 600.0
	PARAMS["MDV0"] = "Random"
	PARAMS["MetaMDBumpHeight"] = 2.0
	PARAMS["MetaMDBumpWidth"] = 3.0
	PARAMS["MetaMaxBumps"] = 2000
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
#	PARAMS["OptMaxCycles"]=500
	web = LocalReactions(EnAndForce,m,50)
	exit(0)
	Opt = MetaOptimizer(EnAndForce,m,Box_=False)
	Opt.MetaOpt(m)

def metaopt_chemsp():
	def qchemdft(m_,basis_ = '6-31g*',xc_='b3lyp', jobtype_='force', filename_='tmp', path_='./qchem/', threads=False):
		istring = '$molecule\n0 1 \n'
		crds = m_.coords.copy()
		crds[abs(crds)<0.0000] *=0.0
		for j in range(len(m_.atoms)):
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
	a=MSet("1,5-hexadiene")
	a.ReadXYZ()
	m=a.mols[0]
	qchemff = lambda x, y: qchemdft(Mol(m.atoms, x), basis_ = '6-311g**',xc_='wb97x-d', jobtype_='force', filename_='tmp', path_='./qchem/', threads=8)
	# TreatedAtoms = np.array([1,6,7,8], dtype=np.uint8)
	# PARAMS["tf_prec"] = "tf.float64"
	# PARAMS["NeuronType"] = "sigmoid_with_param"
	# PARAMS["sigmoid_alpha"] = 100.0
	# PARAMS["HiddenLayers"] = [2000, 2000, 2000]
	# PARAMS["EECutoff"] = 15.0
	# PARAMS["EECutoffOn"] = 0
	# PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	# PARAMS["EECutoffOff"] = 15.0
	# PARAMS["AddEcc"] = True
	# PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]
	# d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
	# tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	# PARAMS["DSFAlpha"] = 0.18*BOHRPERA
	# manager=TFMolManage("chemspider12_nosolvation", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	# def EnAndForce(x_, DoForce=True):
	# 	mtmp = Mol(m.atoms,x_)
	# 	Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
	# 	energy = Etotal[0]
	# 	force = gradient[0]
	# 	if DoForce:
	# 		return energy, force
	# 	else:
	# 		return energy
	web = LocalReactions(qchemff,m,6)


def water_web():
	a=MSet("WebPath")
	a.ReadXYZ()
	TreatedAtoms = a.AtomTypes()
	m=a.mols[0]
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"] = True
	PARAMS["MDMaxStep"] = 50000
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 600.0
	PARAMS["MDV0"] = "Random"
	PARAMS["MetaMDBumpHeight"] = 2.0
	PARAMS["MetaMDBumpWidth"] = 3.0
	PARAMS["MetaMaxBumps"] = 2000
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
	f=open("web_energies.dat", "w")
	for i, mol in enumerate(a.mols):
		en = EnAndForce(mol.coords, DoForce=False)
		f.write(str(i)+"  "+str(en*627.509)+"\n")
	f.close()

def minimize_ob():
	import glob
	import os
	for file in glob.iglob("/media/sdb2/jeherr/tensormol_dev/datasets/chemspider20/uncharged/6*.xyz"):
		try:
			if not os.path.isfile("/media/sdb2/jeherr/tensormol_dev/datasets/chemspider20/uncharged/ob_min"+file[64:-4]+".xyz"):
				mol = Mol()
				mol.read_xyz_with_properties(file, [])
				new_mol = Mol(mol.atoms, ob_minimize_geom(mol))
				new_mol.WriteXYZfile("/media/sdb2/jeherr/tensormol_dev/datasets/chemspider20/uncharged/ob_min", file[64:-4], "w")
		except:
			pass

def run_qchem_meta():
	PARAMS["MDdt"] = 2.0
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 200
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 600.0
	PARAMS["MDV0"] = "Thermal"
	PARAMS["MetaMDBumpHeight"] = 1.00
	PARAMS["MetaMDBumpWidth"] = 2.00
	PARAMS["MetaBumpTime"] = 8.0
	PARAMS["MetaMaxBumps"] = 50
	a = MSet("cs40_li_opt_eq")
	a.Load()
	if len(a.mols) > 100:
		a.cut_max_num_atoms(30)
	mols = random.sample(range(len(a.mols)), min(len(a.mols), 100))
	for i in range(len(mols)):
		try:
			mol = a.mols[i]
			ForceField = lambda x: QchemDFT(Mol(mol.atoms,x),basis_ = '6-311g**',xc_='wB97X-D', jobtype_='force', filename_=mol.properties["name"], path_='./qchem/', threads=8)
			masses = np.array(list(map(lambda x: ATOMICMASSESAMU[x-1],mol.atoms)))
			print("Masses:", masses)
			meta = MetaDynamics(ForceField, mol, EandF_=ForceField)
			meta.Prop()
		except:
			continue

# run_qchem_meta()


# minimize_ob()
# InterpoleGeometries()
# read_unpacked_set()
# TrainKRR(set_="SmallMols_rand", dig_ = "GauSH", OType_="Force")
# RandomSmallSet("master_jeherr2", 1000000)
# TestMetadynamics()
# test_md()
# TestTFBond()
# TestTFGauSH()
# test_gaussian_overlap()
# train_forces_GauSH_direct("SmallMols_rand")
# test_tf_neighbor()
# train_energy_pairs_triples()
# train_energy_symm_func("water_wb97xd_6311gss")
# train_energy_GauSH("water_wb97xd_6311gss")
# train_energy_GauSHv2("chemspider12_wb97xd_6311gss_rand")
train_energy_univ("master_jeherr2_rand")
# eval_test_set_univ("kaggle_opt")
# test_h2o()
# evaluate_BPSymFunc("nicotine_vib")
# water_dimer_plot()
# nicotine_cc_stretch_plot()
# meta_statistics()
# meta_stat_plot()
# harmonic_freq()
# train_Poly_GauSH()
#water_ir()
# GetWaterNetwork()
# water_meta_opt()
# water_meta_react()
# meta_opt()
# metaopt_chemsp()
# water_web()


# PARAMS["tf_prec"] = "tf.float64"
# # PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 12), np.repeat(0.30, 12)), axis=1)
# # PARAMS["SH_NRAD"] = 16
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
# for i in range(100):
# 	b.mols.append(a.mols[i])
# 	# print(b.mols[i].NAtoms())
# maxnatoms = b.MaxNAtom()
# # for mol in b.mols:
# 	# mol.make_neighbors(7.0)
# # max_num_pairs = b.max_neighbors()
#
# zlist = []
# xyzlist = []
# # gradlist = []
# # nnlist = []
# # chargeslist = []
# # n_atoms_list = []
# for i, mol in enumerate(b.mols):
# 	paddedxyz = np.zeros((maxnatoms,3), dtype=np.float64)
# 	paddedxyz[:mol.atoms.shape[0]] = mol.coords
# 	paddedz = np.zeros((maxnatoms), dtype=np.int32)
# 	paddedz[:mol.atoms.shape[0]] = mol.atoms
# 	# paddedcharges = np.zeros((maxnatoms), dtype=np.float64)
# 	# paddedcharges[:mol.atoms.shape[0]] = mol.properties["charges"]
# 	# paddedgrad = np.zeros((maxnatoms,3), dtype=np.float32)
# 	# paddedgrad[:mol.atoms.shape[0]] = mol.properties["gradients"]
# 	# paddednn = np.zeros((maxnatoms, 2), dtype=np.int32)
# 	# paddednn[:mol.atoms.shape[0]] = mol.nearest_ns
# 	# for j, atom_pairs in enumerate(mol.neighbor_list):
# 	# 	molpair = np.stack([np.array([i for _ in range(len(mol.neighbor_list[j]))]), np.array(mol.neighbor_list[j]), mol.atoms[atom_pairs]], axis=-1)
# 	# 	paddedpairs[j,:len(atom_pairs)] = molpair
# 	xyzlist.append(paddedxyz)
# 	zlist.append(paddedz)
# 	# chargeslist.append(paddedcharges)
# 	# gradlist.append(paddedgrad)
# 	# nnlist.append(paddednn)
# 	# n_atoms_list.append(mol.NAtoms())
# 	# if i == 1:
# 	# 	break
# xyzs_tf = tf.cast(tf.stack(xyzlist), tf.float64)
# zs_tf = tf.cast(tf.stack(zlist), tf.int32)
# # charges_tf = tf.cast(tf.stack(chargeslist), tf.float32)
# xyzs_np = np.stack(xyzlist).astype(np.float64)
# zs_np = np.stack(zlist).astype(np.int32)
# # gradstack = tf.stack(gradlist)
# # nnstack = tf.stack(nnlist)
# # natomsstack = tf.stack(n_atoms_list)
# # r_cutoff = 6.5
# # gauss_params = tf.Variable(PARAMS["RBFS"], trainable=True, dtype=tf.float32)
# # elements = [1, 6, 7, 8]
# # elements_tf = tf.constant([1, 6, 7, 8], dtype=tf.int32)
# # element_pairs = np.array([[elements[i], elements[j]] for i in range(len(elements)) for j in range(i, len(elements))])
# # element_pairs_tf = tf.constant(element_pairs, dtype=tf.int32)
#
# element_codes = tf.Variable(ELEMENTCODES, trainable=False, dtype=tf.float64)
# # element_codepairs = np.zeros((int(ELEMENTCODES.shape[0]*(ELEMENTCODES.shape[0]+1)/2), ELEMENTCODES.shape[1]))
# # codepair_idx = np.zeros((ELEMENTCODES.shape[0], ELEMENTCODES.shape[0]), dtype=np.int32)
# # counter = 0
# # for i in range(len(ELEMENTCODES)):
# # 	for j in range(i, len(ELEMENTCODES)):
# # 		codepair_idx[i,j] = counter
# # 		codepair_idx[j,i] = counter
# # 		element_codepairs[counter] = ELEMENTCODES[i] * ELEMENTCODES[j]
# # 		counter += 1
# # element_codepairs_tf = tf.Variable(element_codepairs, trainable=False, dtype=tf.float32)
# # codepair_idx_tf = tf.Variable(codepair_idx, trainable=False, dtype=tf.int32)
#
# eta = PARAMS["AN1_eta"]
# zeta = PARAMS["AN1_zeta"]
#
# #Define radial grid parameters
# num_radial_rs = PARAMS["AN1_num_r_Rs"]
# radial_cutoff = PARAMS["AN1_r_Rc"]
# radial_rs = radial_cutoff * np.linspace(0, (num_radial_rs - 1.0) / num_radial_rs, num_radial_rs)
#
# #Define angular grid parameters
# num_angular_rs = PARAMS["AN1_num_a_Rs"]
# num_angular_theta_s = PARAMS["AN1_num_a_As"]
# angular_cutoff = PARAMS["AN1_a_Rc"]
# theta_s = np.pi * np.linspace(0, (num_angular_theta_s - 1.0) / num_angular_theta_s, num_angular_theta_s)
# angular_rs = angular_cutoff * np.linspace(0, (num_angular_rs - 1.0) / num_angular_rs, num_angular_rs)
#
# radial_rs_tf = tf.Variable(radial_rs, trainable=False, dtype = tf.float64)
# angular_rs_tf = tf.Variable(angular_rs, trainable=False, dtype = tf.float64)
# theta_s_tf = tf.Variable(theta_s, trainable=False, dtype = tf.float64)
# radial_cutoff_tf = tf.Variable(radial_cutoff, trainable=False, dtype = tf.float64)
# angular_cutoff_tf = tf.Variable(angular_cutoff, trainable=False, dtype = tf.float64)
# zeta_tf = tf.Variable(zeta, trainable=False, dtype = tf.float64)
# eta_tf = tf.Variable(eta, trainable=False, dtype = tf.float64)
#
# nlt = MolEmb.Make_NLTensor(xyzs_np, zs_np, radial_cutoff, maxnatoms, True, True)
# tlt = MolEmb.Make_TLTensor(xyzs_np, zs_np, angular_cutoff, maxnatoms, False)
# nlt_tf = tf.constant(nlt, dtype=tf.int32)
# tlt_tf = tf.constant(tlt, dtype=tf.int32)
# # replace_idx = tf.constant([0, 2], dtype=tf.int32)
# # replace_codes = tf.Variable(ELEMENTCODES[15], trainable=False, dtype=tf.float32)
# # gather_replace_codepairs = codepair_idx_tf[15]
# # replace_codepairs = tf.gather(element_codepairs_tf, gather_replace_codepairs)
# tmp = tf_sym_func_element_codes(xyzs_tf, zs_tf, nlt_tf, tlt_tf, element_codes, radial_rs_tf,
# 		radial_cutoff_tf, angular_rs_tf, theta_s_tf, angular_cutoff_tf, zeta_tf, eta_tf)
# # tmp2 = tf_sym_func_element_codes_v3(xyzs_tf, zs_tf, nlt_tf, tlt_tf, element_codes, radial_rs_tf,
# # 		radial_cutoff_tf, angular_rs_tf, theta_s_tf, angular_cutoff_tf, zeta_tf, eta_tf)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()
# @TMTiming("test")
# def get_pairs():
# 	tmp3 = sess.run(tmp, options=options, run_metadata=run_metadata)
# 	return tmp3
# tmp5 = get_pairs()
# print(tmp5)
# print(tmp5.shape)
# fetched_timeline = timeline.Timeline(run_metadata.step_stats)
# chrome_trace = fetched_timeline.generate_chrome_trace_format()
# with open('timeline_step_tmp_tm_nocheck_h2o.json', 'w') as f:
# 	f.write(chrome_trace)

# def gather_coul(xyzs, Zs, atom_charges, pairs):
# 	padding_mask = tf.where(tf.logical_and(tf.not_equal(Zs, 0), tf.reduce_any(tf.not_equal(pairs, -1), axis=-1)))
# 	central_atom_coords = tf.gather_nd(xyzs, padding_mask)
# 	central_atom_charge = tf.gather_nd(atom_charges, padding_mask)
# 	pairs = tf.gather_nd(pairs, padding_mask)
# 	padded_pairs = tf.equal(pairs, -1)
# 	tmp_pairs = tf.where(padded_pairs, tf.zeros_like(pairs), pairs)
# 	gather_pairs = tf.stack([tf.cast(tf.tile(padding_mask[:,:1], [1, tf.shape(pairs)[1]]), tf.int32), tmp_pairs], axis=-1)
# 	pair_coords = tf.gather_nd(xyzs, gather_pairs)
# 	dxyzs = tf.expand_dims(central_atom_coords, axis=1) - pair_coords
# 	pair_mask = tf.where(padded_pairs, tf.zeros_like(pairs), tf.ones_like(pairs))
# 	dxyzs *= tf.cast(tf.expand_dims(pair_mask, axis=-1), eval(PARAMS["tf_prec"]))
# 	pair_charges = tf.gather_nd(atom_charges, gather_pairs)
# 	pair_charges *= tf.cast(pair_mask, eval(PARAMS["tf_prec"]))
# 	q1q2 = tf.expand_dims(central_atom_charge, axis=-1) * pair_charges
# 	return dxyzs, q1q2, padding_mask
#
# def calculate_coul_energy(dxyzs, q1q2, scatter_idx, max_num_atoms):
# 	"""
# 	Polynomial cutoff 1/r (in BOHR) obeying:
# 	kern = 1/r at SROuter and LRInner
# 	d(kern) = d(1/r) (true force) at SROuter,LRInner
# 	d**2(kern) = d**2(1/r) at SROuter and LRInner.
# 	d(kern) = 0 (no force) at/beyond SRInner and LROuter
#
# 	The hard cutoff is LROuter
# 	"""
# 	srange_inner = tf.constant(6.0*1.889725989, dtype=tf.float64)
# 	srange_outer = tf.constant(9.0*1.889725989, dtype=tf.float64)
# 	lrange_inner = tf.constant(13.0*1.889725989, dtype=tf.float64)
# 	lrange_outer = tf.constant(15.0*1.889725989, dtype=tf.float64)
# 	a, b, c, d, e, f, g, h = -43.568, 15.9138, -2.42286, 0.203849, -0.0102346, 0.000306595, -5.0738e-6, 3.57816e-8
# 	# a, b, c, d, e, f, g, h = -43.568, 30.0728, -8.65219, 1.37564, -0.130517, 0.00738856, -0.000231061, 3.0793e-6
# 	#a, b, c, d, e, f, g, h = -12.8001, 10.2348, -3.21999, 0.556841, -0.0571471, 0.0034799, -0.000116418, 1.65087e-6
# 	dist = tf.norm(dxyzs+1.e-16, axis=-1)
# 	dist *= 1.889725989
# 	dist = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) * srange_inner, dist)
# 	dist = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) * lrange_outer, dist)
# 	dist2 = dist * dist
# 	dist3 = dist2 * dist
# 	dist4 = dist3 * dist
# 	dist5 = dist4 * dist
# 	dist6 = dist5 * dist
# 	dist7 = dist6 * dist
# 	kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
# 	# kern = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) / srange_inner, mrange_kern)
# 	# kern = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) / lrange_outer, kern)
# 	mrange_energy = tf.reduce_sum(kern * q1q2, axis=1)
# 	lrange_energy = tf.reduce_sum(q1q2, axis=1) / lrange_outer
# 	coulomb_energy = mrange_energy - lrange_energy
# 	return tf.reduce_sum(tf.scatter_nd(scatter_idx, coulomb_energy, [1000, max_num_atoms]), axis=-1) / 2.0
#
# def wrapper(xyzs, Zs, atom_charges, pairs, maxnatoms):
# 	dxyzs, q1q2, scatter_idx = gather_coul(xyzs, Zs, atom_charges, pairs)
# 	coul_e = calculate_coul_energy(dxyzs, q1q2, scatter_idx, maxnatoms)
# 	return q1q2
#
#
#
# ms = MSet("kaggle_opt")
# ms.Load()
# maxnatoms = ms.MaxNAtom()
#
# zlist = []
# xyzlist = []
# gradlist = []
# nnlist = []
# chargeslist = []
# energylist = []
# # n_atoms_list = []
# for i, mol in enumerate(ms.mols):
# 	paddedxyz = np.zeros((maxnatoms,3), dtype=np.float64)
# 	paddedxyz[:mol.atoms.shape[0]] = mol.coords
# 	paddedz = np.zeros((maxnatoms), dtype=np.int32)
# 	paddedz[:mol.atoms.shape[0]] = mol.atoms
# 	paddedcharges = np.zeros((maxnatoms), dtype=np.float64)
# 	paddedcharges[:mol.atoms.shape[0]] = mol.properties["charges"]
# 	# paddedgrad = np.zeros((maxnatoms,3), dtype=np.float32)
# 	# paddedgrad[:mol.atoms.shape[0]] = mol.properties["gradients"]
# 	# paddednn = np.zeros((maxnatoms, 2), dtype=np.int32)
# 	# paddednn[:mol.atoms.shape[0]] = mol.nearest_ns
# 	# for j, atom_pairs in enumerate(mol.neighbor_list):
# 	# 	molpair = np.stack([np.array([i for _ in range(len(mol.neighbor_list[j]))]), np.array(mol.neighbor_list[j]), mol.atoms[atom_pairs]], axis=-1)
# 	# 	paddedpairs[j,:len(atom_pairs)] = molpair
# 	xyzlist.append(paddedxyz)
# 	zlist.append(paddedz)
# 	chargeslist.append(paddedcharges)
# 	energylist.append(mol.properties["energy"])
# 	# gradlist.append(paddedgrad)
# 	# nnlist.append(paddednn)
# 	# n_atoms_list.append(mol.NAtoms())
# 	# if i == 1:
# 	# 	break
# xyzs_np = np.stack(xyzlist).astype(np.float64)
# zs_np = np.stack(zlist).astype(np.int32)
# charges_np = np.stack(chargeslist).astype(np.float64)
# energy_np = np.stack(energylist).astype(np.float64)

# xyzs_pl = tf.placeholder(tf.float64, shape=[1000, maxnatoms, 3])
# Zs_pl = tf.placeholder(tf.int32, shape=[1000, maxnatoms])
# charges_pl = tf.placeholder(tf.float64, shape=[1000, maxnatoms])
# coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[1000, maxnatoms, None])
#
# coulomb_e = wrapper(xyzs_pl, Zs_pl, charges_pl, coulomb_pairs_pl, maxnatoms)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for i in range(int(len(zs_np) / 1000.)):
# 	coulomb_pairs = MolEmb.Make_NLTensor(xyzs_np[i*1000:(i+1)*1000], zs_np[i*1000:(i+1)*1000], 19.0, maxnatoms, True, False)
# 	feed_dict = {xyzs_pl:xyzs_np[i*1000:(i+1)*1000], Zs_pl:zs_np[i*1000:(i+1)*1000], charges_pl:charges_np[i*1000:(i+1)*1000],
# 				coulomb_pairs_pl:coulomb_pairs}
# 	ce = sess.run(coulomb_e, feed_dict=feed_dict)
# 	print(ce[0])
# 	exit(0)
# 	for j in range(1000):
# 		a.mols[(i*1000)+j].properties["coulomb_kern_energy"] = ce[j]
# a.Save("kaggle_opt_tmp")
#
# ms.mols[0].BuildDistanceMatrix()
# ms.mols[0].coulomb_matrix = ms.mols[0].properties["charges"] * np.expand_dims(ms.mols[0].properties["charges"], axis=-1)
# for i in range(len(ms.mols[0].coulomb_matrix)):
# 	ms.mols[0].coulomb_matrix[i,i] = 0.0
# srange_inner = 6.0*1.889725989
# srange_outer = 9.0*1.889725989
# lrange_inner = 13.0*1.889725989
# lrange_outer = 15.0*1.889725989
# a, b, c, d, e, f, g, h = -43.568, 15.9138, -2.42286, 0.203849, -0.0102346, 0.000306595, -5.0738e-6, 3.57816e-8
#
# dist = ms.mols[0].DistMatrix * 1.889725989
# dist = np.where(np.less(dist, srange_inner), np.ones_like(dist) * srange_inner, dist)
# dist = np.where(np.greater(dist, lrange_outer), np.ones_like(dist) * lrange_outer, dist)
# dist2 = dist * dist
# dist3 = dist2 * dist
# dist4 = dist3 * dist
# dist5 = dist4 * dist
# dist6 = dist5 * dist
# dist7 = dist6 * dist
# kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
# # kern = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) / srange_inner, mrange_kern)
# # kern = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) / lrange_outer, kern)
# mrange_energy = np.sum(kern * ms.mols[0].coulomb_matrix, axis=1)
# lrange_energy = np.sum(ms.mols[0].coulomb_matrix, axis=1) / lrange_outer
# coulomb_energy = mrange_energy - lrange_energy
# print(np.sum(coulomb_energy) / 2.0)

# a=MSet("cs40_b_opt")
# a.Load()
# b=MSet("cs40_b_opt_eq")
# names = []
# for mol in a.mols:
# 	if "name" in mol.properties:
# 		names.append(mol.properties["name"])
# names = list(set(names))
# for name in names:
# 	eq_mol = Mol()
# 	eq_mol.properties["energy"] = 0.0
# 	for mol in a.mols:
# 		if "name" in mol.properties:
# 			if mol.properties["name"] == name:
# 				if mol.properties["energy"] < eq_mol.properties["energy"]:
# 					eq_mol = mol
# 	b.mols.append(eq_mol)
# print(len(b.mols))
# b.Save()

#a=MSet("cs40_li_opt")
#a.read_xyz_set_with_properties(path="/media/sdb2/jeherr/tensormol_dev/datasets/chemspider40/uncharged/opt/li/data/", properties=["name", "energy", "gradients", "dipole", "charges"])
#print(len(a.mols), " Molecules")
#a.Save()
