from TensorMol import *

def evaluate_set(mset):
	"""
	Evaluate energy, force, and charge error statistics on an entire set
	using the Symmetry Function Universal network. Prints MAE, MSE, and RMSE.
	"""
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Tue_May_15_10.18.25_2018")
	molset = MSet(mset)
	molset.Load()
	energy_errors, gradient_errors, charge_errors = network.evaluate_set(molset)
	print(energy_errors[:10])
	print(gradient_errors[:10])
	print(charge_errors[:10])
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

# evaluate_set("kaggle_opt")

def evaluate_mol(mol):
	"""
	Evaluate single point energy, force, and charge for a molecule using the
	Symmetry Function Universal network.
	"""
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Tue_May_15_10.18.25_2018")
	energy, forces, charges = network.evaluate_mol(mol)
	print("Energy label: ", mol.properties["energy"], " Prediction: ", energy)
	print("Force labels: ", -mol.properties["gradients"], " Prediction: ", forces)
	print("Charge label: ", mol.properties["charges"], " Prediction: ", charges)

# a=MSet("kaggle_opt")
# a.Load()
# mol=a.mols[0]
# evaluate_mol(mol)

def run_md(mol):
	"""
	Run a molecular dynamics simulation using the Symmetry Function Universal network.
	"""
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 20000
	PARAMS["MDThermostat"] = None
	PARAMS["MDTemp"]= 300.0
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Tue_May_15_10.18.25_2018")
	def force_field(coords, eval_forces=True):
		m=Mol(mol.atoms, coords)
		energy, forces, charges = network.evaluate_mol(m)
		if eval_forces:
			return energy, JOULEPERHARTREE*forces
		else:
			return energy
	md = VelocityVerlet(force_field, mol, EandF_=force_field)
	md.Prop()

a=MSet("kaggle_opt")
a.Load()
mol=a.mols[0]
run_md(mol)

def TestKaggle():
	a=MSet("kaggle_opt")
	a.Load()
	mol=a.mols[0]
	evaluate_mol(mol)

TestKaggle()

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
