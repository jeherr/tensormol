from TensorMol import *

def evaluate_set(mset):
	"""
	Evaluate energy, force, and charge error statistics on an entire set
	using the Symmetry Function Universal network. Prints MAE, MSE, and RMSE.
	"""
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Thu_May_31_16.20.05_2018")
	molset = MSet(mset)
	molset.Load()
	energy_errors, gradient_errors = network.evaluate_set(molset)
	mae_e = np.mean(np.abs(energy_errors))
	mse_e = np.mean(energy_errors)
	rmse_e = np.sqrt(np.mean(np.square(energy_errors)))
	mae_g = np.mean(np.abs(gradient_errors))
	mse_g = np.mean(gradient_errors)
	rmse_g = np.sqrt(np.mean(np.square(gradient_errors)))
	# mae_c = np.mean(np.abs(charge_errors))
	# mse_c = np.mean(charge_errors)
	# rmse_c = np.sqrt(np.mean(np.square(charge_errors)))
	print("MAE  Energy: ", mae_e, " Gradients: ", mae_g)#, " Charges: ", mae_c)
	print("MSE  Energy: ", mse_e, " Gradients: ", mse_g)#, " Charges: ", mse_c)
	print("RMSE  Energy: ", rmse_e, " Gradients: ", rmse_g)#, " Charges: ", rmse_c)

# evaluate_set("kaggle_opt")

def evaluate_mol(mol):
	"""
	Evaluate single point energy, force, and charge for a molecule using the
	Symmetry Function Universal network.
	"""
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Tue_May_15_10.18.25_2018")
	energy, forces, charges = network.evaluate_mol(mol)
	print(energy, forces, charges)
	# print("Energy label: ", mol.properties["energy"], " Prediction: ", energy)
	# print("Force labels: ", -mol.properties["gradients"], " Prediction: ", forces)
	# print("Charge label: ", mol.properties["charges"], " Prediction: ", charges)

# a=MSet("methanol")
# a.ReadXYZ()
# mol=a.mols[0]
# evaluate_mol(mol)

def run_alchemical_trans(mols):
	"""
	Run an alchemical transformation using the Symmetry Function Universal network.

	Args:
		mols (list): a sequential list of Mols to undergo the alchemical changes
	"""
	PARAMS["MetaBowlK"] = 0.05
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 10000
	PARAMS["MDThermostat"] = None
	PARAMS["MDTemp"]= 300.0
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Thu_May_31_16.20.05_2018")
	alchem_ef = network.evaluate_alchem_mol(mols)
	transitions = [1,1000]
	alchemical_transformation(alchem_ef, mols, mols[0].coords, transitions)

# a=MSet("diazepam", center_=False)
# a.ReadXYZ()
# b=MSet("water_diazepam", center_=False)
# b.ReadXYZ()
# mols = [a.mols[0], b.mols[0]]
# run_alchemical_trans(mols)

def run_element_opt(mol):
	mol.atoms[18] = 7
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Thu_May_31_16.20.05_2018")
	replace_idxs = [[0,18]]
	# network.get_element_opt_function(mol, replace_idxs)
	EF, original_codes, original_codepairs = network.get_element_opt_function(mol, replace_idxs)
	energy, forces, codes_force, codepairs_force = EF(mol.coords, original_codes, original_codepairs)
	atom_codes = original_codes + codes_force
	atom_codepairs = original_codepairs + codepairs_force
	for i in range(5000):
		energy, forces, codes_force, codepairs_force = EF(mol.coords, atom_codes, atom_codepairs)
		atom_codes += codes_force
		atom_codepairs += codepairs_force
		rms_codeforce = np.sqrt(np.mean(np.square(codes_force)))
		rms_codepairforce = np.sqrt(np.mean(np.square(codepairs_force)))
		print(energy, rms_codeforce, rms_codepairforce)
	print(atom_codes)


a=MSet("diazepam")
a.ReadXYZ()
run_element_opt(a.mols[0])

def alchem_chain_opt(mols):
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Thu_May_31_16.20.05_2018")
	alchem_ef = network.evaluate_alchem_mol(mols)
	Opt = AlchemGeomOptimizer(alchem_ef)
	for i in range(21):
		mols=Opt.Opt(mols, i*0.05, "alchem_opt", eff_max_step=10000)

# a=MSet("etoh_dimer", center_=False)
# a.ReadXYZ()
# b=MSet("etoh_dimer_to_water", center_=False)
# b.ReadXYZ()
# mols = [a.mols[0], b.mols[0]]
# alchem_chain_opt(mols)


def run_md(mol):
	"""
	Run a molecular dynamics simulation using the Symmetry Function Universal network.
	"""
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 20000
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDTemp"]= 300.0
	network = UniversalNetwork(name="SF_Universal_master_jeherr_Thu_May_31_20.09.30_2018")
	EF = network.GetEnergyForceRoutine(mol)
	md = VelocityVerlet(EF, mol, EandF_=EF)
	md.Prop()

# a=MSet("diazepam_GABA", center_=False)
# a.ReadXYZ()
# mol=a.mols[0]
# run_md(mol)



def TestKaggle():
	a=MSet("kaggle_opt")
	a.Load()
	mol=a.mols[0]
	evaluate_mol(mol)

def TestOpt():
	a=MSet("phosphoric_acid")
	a.ReadXYZ()
	m=a.mols[0]
	# m = Mol()
	# m.FromXYZString("""73
	#
	# O          3.11000        4.22490       -0.75810
	# O          4.72290        2.06780       -2.02160
	# O          3.28790        0.27660       -2.12830
	# O          7.57740       -2.18410        0.83530
	# O          6.93870       -0.24500        1.85400
	# N         -0.44900        1.57680        0.54520
	# N          0.67240       -1.09000        0.16920
	# N         -3.08650        0.73580        0.30880
	# N         -2.08930       -2.17120        0.36140
	# C          3.15530        1.80910       -0.26920
	# C          1.94310        0.99350        0.14610
	# C          1.10470        3.11900       -0.07220
	# C          0.85730        1.76100        0.25120
	# C          2.53600        3.20940       -0.40610
	# C         -0.01170        3.83890       -0.00490
	# C          1.80260       -0.45930        0.35870
	# C         -0.97430        2.76340        0.37660
	# C          2.91740       -1.33270        0.77810
	# C          2.32400       -2.53760        0.79670
	# C          3.70160        1.28460       -1.55560
	# C          0.92000       -2.41280        0.43150
	# C         -2.41080        3.07580        0.55540
	# C         -0.29110        5.26060       -0.27150
	# C         -3.30810        2.07470        0.53020
	# C         -4.22430       -0.01890        0.37480
	# C         -1.33840       -3.26350       -0.02560
	# C          0.02500       -3.41140        0.34800
	# C         -4.76240        2.20220        0.74210
	# C         -5.29980        0.95570        0.65430
	# C         -3.38890       -2.29630       -0.08630
	# C         -2.18770       -4.11130       -0.70980
	# C         -3.46520       -3.50710       -0.75000
	# C          4.24800       -0.96910        1.13640
	# C         -4.40070       -1.34110        0.20870
	# C          2.93270       -3.85050        1.16890
	# C         -6.72700        0.53540        0.79750
	# C         -1.82240       -5.42330       -1.29120
	# C         -5.50430        3.40910        1.00050
	# C         -4.63100       -4.03780       -1.35240
	# C          5.32530       -1.71620        0.83820
	# C          5.31710        1.65030       -3.25560
	# C         -6.03270        4.16680        0.03880
	# C         -5.68440       -3.34470       -1.89440
	# C          6.67040       -1.26750        1.24620
	# H          3.91490        1.82320        0.51520
	# H         -2.69930        4.10740        0.71860
	# H         -0.66660        5.75450        0.62990
	# H          0.61110        5.79090       -0.59250
	# H         -1.03960        5.36580       -1.06280
	# H          0.35400       -4.43150        0.53040
	# H         -5.40880       -1.73500        0.30730
	# H          4.36030       -0.04870        1.70090
	# H          3.88280       -3.75330        1.69880
	# H          2.27760       -4.40220        1.85250
	# H          3.09460       -4.46420        0.27690
	# H         -6.83900       -0.17840        1.62010
	# H         -7.08680        0.06850       -0.12540
	# H         -7.38530        1.38280        1.01220
	# H         -2.13640       -6.23380       -0.62560
	# H         -2.30250       -5.57200       -2.26410
	# H         -0.74330       -5.51440       -1.45110
	# H         -5.62320        3.69670        2.04090
	# H         -4.70070       -5.12240       -1.41710
	# H          5.25800       -2.63030        0.25800
	# H          4.57150        1.65850       -4.05620
	# H          5.75710        0.65460       -3.14550
	# H          6.11050        2.35890       -3.50730
	# H         -6.58170        5.06770        0.29090
	# H         -5.93330        3.91260       -1.01130
	# H         -6.51300       -3.89300       -2.33170
	# H         -5.71990       -2.26400       -1.94770
	# H          8.49250       -1.93230        1.08330
	# Mg        -1.34673        0.02041       -0.06327""")
	net = UniversalNetwork(name="SF_Universal_master_jeherr_Thu_May_31_20.09.30_2018")
	EF = net.GetEnergyForceRoutine(m)
	Opt = GeomOptimizer(EF)
	m1=Opt.Opt(m,"TEST",eff_max_step=5000)

# TestOpt()

def TestBernyOpt():
	from berny import Berny, geomlib
	# a=MSet("kaggle_opt")
	# a.Load()
	# m=a.mols[8804]
	# m = Mol()
	# m.FromXYZString("""73
	xyz = """73

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
	Mg        -1.34673        0.02041       -0.06327"""

	init_mol = Mol()
	init_mol.FromXYZString(xyz)
	# init_mol = a.mols[0]
	net = UniversalNetwork(name="SF_Universal_master_jeherr_Thu_May_31_16.20.05_2018")
	EF = net.GetEnergyForceRoutine(init_mol)
	optimizer = Berny(
	geomlib.loads(init_mol.__str__(), 'xyz'),
		maxsteps=5000)
		# gradientrms=0.0001,
		# gradientmax=0.003,
		# debug=True)
	debug = []
	last_geom = ""
	init_mol.WriteXYZfile(fname="chlorophyll_berny")
	for geom in optimizer:
		molecule_xyz = geom.dumps('xyz')
		molecule = Mol()
		molecule.FromXYZString(molecule_xyz)
		molecule.WriteXYZfile(fname="chlorophyll_berny")
		energy, gradients = EF(molecule.coords)  # calculate energy and gradients of geom
		info = optimizer.send((energy, gradients))
		debug.append(info)
	print('Final Geometry:')
	print(geom.dumps('xyz'))

# TestBernyOpt()

def TestNeb():
	net = UniversalNetwork(name="SF_Universal_master_jeherr_Tue_May_15_10.18.25_2018")
	s = MSet()
	s.ReadXYZ("Endiandric")
	NEB = BatchedNudgedElasticBand(net,s.mols[0],s.mols[3],thresh_=0.003,nbeads_=20)
	NEB.Opt(eff_max_step=100)

#TestNeb()

# m = Mol()
# m.FromXYZString("""72
#
# O          3.11000        4.22490       -0.75810
# O          4.72290        2.06780       -2.02160
# O          3.28790        0.27660       -2.12830
# O          7.57740       -2.18410        0.83530
# O          6.93870       -0.24500        1.85400
# N         -0.44900        1.57680        0.54520
# N          0.67240       -1.09000        0.16920
# N         -3.08650        0.73580        0.30880
# N         -2.08930       -2.17120        0.36140
# C          3.15530        1.80910       -0.26920
# C          1.94310        0.99350        0.14610
# C          1.10470        3.11900       -0.07220
# C          0.85730        1.76100        0.25120
# C          2.53600        3.20940       -0.40610
# C         -0.01170        3.83890       -0.00490
# C          1.80260       -0.45930        0.35870
# C         -0.97430        2.76340        0.37660
# C          2.91740       -1.33270        0.77810
# C          2.32400       -2.53760        0.79670
# C          3.70160        1.28460       -1.55560
# C          0.92000       -2.41280        0.43150
# C         -2.41080        3.07580        0.55540
# C         -0.29110        5.26060       -0.27150
# C         -3.30810        2.07470        0.53020
# C         -4.22430       -0.01890        0.37480
# C         -1.33840       -3.26350       -0.02560
# C          0.02500       -3.41140        0.34800
# C         -4.76240        2.20220        0.74210
# C         -5.29980        0.95570        0.65430
# C         -3.38890       -2.29630       -0.08630
# C         -2.18770       -4.11130       -0.70980
# C         -3.46520       -3.50710       -0.75000
# C          4.24800       -0.96910        1.13640
# C         -4.40070       -1.34110        0.20870
# C          2.93270       -3.85050        1.16890
# C         -6.72700        0.53540        0.79750
# C         -1.82240       -5.42330       -1.29120
# C         -5.50430        3.40910        1.00050
# C         -4.63100       -4.03780       -1.35240
# C          5.32530       -1.71620        0.83820
# C          5.31710        1.65030       -3.25560
# C         -6.03270        4.16680        0.03880
# C         -5.68440       -3.34470       -1.89440
# C          6.67040       -1.26750        1.24620
# H          3.91490        1.82320        0.51520
# H         -2.69930        4.10740        0.71860
# H         -0.66660        5.75450        0.62990
# H          0.61110        5.79090       -0.59250
# H         -1.03960        5.36580       -1.06280
# H          0.35400       -4.43150        0.53040
# H         -5.40880       -1.73500        0.30730
# H          4.36030       -0.04870        1.70090
# H          3.88280       -3.75330        1.69880
# H          2.27760       -4.40220        1.85250
# H          3.09460       -4.46420        0.27690
# H         -6.83900       -0.17840        1.62010
# H         -7.08680        0.06850       -0.12540
# H         -7.38530        1.38280        1.01220
# H         -2.13640       -6.23380       -0.62560
# H         -2.30250       -5.57200       -2.26410
# H         -0.74330       -5.51440       -1.45110
# H         -5.62320        3.69670        2.04090
# H         -4.70070       -5.12240       -1.41710
# H          5.25800       -2.63030        0.25800
# H          4.57150        1.65850       -4.05620
# H          5.75710        0.65460       -3.14550
# H          6.11050        2.35890       -3.50730
# H         -6.58170        5.06770        0.29090
# H         -5.93330        3.91260       -1.01130
# H         -6.51300       -3.89300       -2.33170
# H         -5.71990       -2.26400       -1.94770
# H          8.49250       -1.93230        1.08330
# """)
# network = UniversalNetwork(name="SF_Universal_master_jeherr_Thu_May_31_20.09.30_2018")
# vec = m.coords[41] - m.coords[37]
# for i in range(-50,50):
# 	mol = Mol(m.atoms, m.coords)
# 	mol.coords[41] += i*0.005*vec
# 	mol.WriteXYZfile()
# 	energy, atme, nne, ce, forces, charges = network.evaluate_mol(mol)
# 	print(ce)
