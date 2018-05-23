from TensorMol import *

def evaluate_set(mset):
	"""
	Evaluate energy, force, and charge error statistics on an entire set
	using the Symmetry Function Universal network. Prints MAE, MSE, and RMSE.
	"""
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

# evaluate_set("kaggle_opt")

def evaluate_mol(mol):
	"""
	Evaluate single point energy, force, and charge for a molecule using the
	Symmetry Function Universal network.
	"""
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
	energy, forces, charges = network.evaluate_mol(mol)
	print("Energy label: ", mol.properties["energy"], " Prediction: ", energy)
	print("Force labels: ", -mol.properties["gradients"], " Prediction: ", forces)
	print("Charge label: ", mol.properties["charges"], " Prediction: ", charges)

a=MSet("kaggle_opt")
a.Load()
mol=a.mols[0]
evaluate_mol(mol)
