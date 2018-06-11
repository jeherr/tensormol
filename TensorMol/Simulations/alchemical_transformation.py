"""
The Units chosen are Angstrom, Fs.
I convert the force outside from kcal/(mol angstrom) to Joules/(mol angstrom)
"""

from __future__ import absolute_import
from __future__ import print_function
from ..Containers.Sets import *
from .SimpleMD import *
from ..ForceModels.Electrostatics import *
from ..Math.QuasiNewtonTools import *
from ..Math.Statistics import *

def write_trajectory(mol, name):
	mol.WriteXYZfile("./results/", "MDTrajectory"+name)
	return

def initialize_velocities(coords, masses, temp):
	if PARAMS["MDV0"]=="Random":
		LOGGER.info("Using random initial velocities.")
		np.random.seed()
		veloc = np.random.randn(*coords.shape)
		# Do this elementwise otherwise H's blow off.
		for i in range(len(masses[0])):
			if masses[0,i] == 0.0:
				atom_mass = masses[1,i]
			else:
				atom_mass = masses[0,i]
			effective_temp = ((2.0 / (3.0 * IDEALGASR)) * pow(10.0,10.0) * (1./2.)
							* atom_mass * np.einsum("i,i", veloc[i], veloc[i]))
			if effective_temp != 0.0:
				veloc[i] *= np.sqrt(temp / effective_temp)
	elif PARAMS["MDV0"]=="Thermal":
		LOGGER.info("Using thermal initial velocities.")
		veloc = np.random.normal(size=coords.shape[0]) * np.sqrt(1.38064852e-23 * temp / masses)[:,None]
	return veloc

def get_thermostat(masses, veloc):
	if (PARAMS["MDThermostat"]=="Rescaling"):
		LOGGER.info("Using Rescaling thermostat.")
		thermostat = Thermostat(masses, veloc)
	elif (PARAMS["MDThermostat"]=="Nose"):
		LOGGER.info("Using Nose thermostat.")
		thermostat = NoseThermostatAlchem(masses, veloc)
	elif (PARAMS["MDThermostat"]=="Andersen"):
		LOGGER.info("Using Andersen thermostat.")
		thermostat = AndersenThermostatAlchem(masses, veloc)
	elif (PARAMS["MDThermostat"]=="Langevin"):
		LOGGER.info("Using Langevin thermostat.")
		thermostat = LangevinThermostat(masses, veloc)
	elif (PARAMS["MDThermostat"]=="NoseHooverChain"):
		LOGGER.info("Using NoseHooverChain thermostat.")
		thermostat = NoseChainThermostat(masses, veloc)
	else:
		LOGGER.info("No thermostat chosen. Performing unthermostatted MD.")
		thermostat = None
	return thermostat

def velocity_verlet_step(force_field, mols, coords, veloc, accel, masses, delta, dt):
	"""
	A velocity verlet step for an alchemical transformation MD

	Args:
		force_field: The force function (returns Joules/Angstrom)
		coords: Current coordinates (A)
		veloc: Velocities (A/fs)
		accel: The acceleration at current step. (A^2/fs^2)
		masses: the mass vector. (kg)
		dt: time step (fs)
	Returns:
		x: updated positions
		v: updated Velocities
		a: updated accelerations
		e: Energy at midpoint.
	"""
	coords = coords + veloc * dt + (1./2.) * accel * dt * dt
	for mol in mols:
		mol.coords = coords[:mol.NAtoms()]
	potential, forces = force_field(mols, delta)
	forces *= JOULEPERHARTREE
	new_accel = pow(10.0,-10.0) * np.einsum("ax,a->ax", forces, 1.0 / masses)
	new_veloc = veloc + (1./2.) * (accel + new_accel) * dt
	return mols, coords, new_veloc, new_accel, potential, forces

class NoseThermostatAlchem(Thermostat):
	def __init__(self, masses, init_veloc):
		"""
		Velocity Verlet step with a Nose-Hoover Thermostat.
		"""
		self.m = masses.copy()
		self.N = len(masses)
		self.T = PARAMS["MDTemp"]  # Length of NH chain.
		self.eta = 0.0
		self.name = "Nose"
		self.Rescale(init_veloc)
		print("Using ", self.name, " thermostat at ", self.T, " degrees Kelvin")
		return

	def step(self, force_field, mols, coords, veloc, accel, masses, delta, dt):
		"""
		http://www2.ph.ed.ac.uk/~dmarendu/MVP/MVP03.pdf
		"""
		# Recompute these stepwise in case of variable T.
		self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.tau = 20.0*PARAMS["MDdt"]*self.N
		self.Q = self.kT*self.tau*self.tau
		coords = coords + veloc * dt + (1./2.) * (accel - self.eta * veloc) * dt * dt
		for mol in mols:
			mol.coords = coords[:mol.NAtoms()]
		vdto2 = veloc + (1./2.) * (accel - self.eta * veloc) * dt
		potential, forces = force_field(mols, delta)
		forces *= JOULEPERHARTREE
		new_accel = pow(10.0,-10.0) * np.einsum("ax,a->ax", forces, 1.0 / masses) # m^2/s^2 => A^2/Fs^2
		kinetic = (1./2.) * np.dot(np.einsum("ia,ia->i",veloc, veloc), masses)
		etadto2 = self.eta + (dt / (2.0 * self.Q)) * (kinetic - (((3.0 * self.N + 1) / 2.0)) * self.kT)
		kedto2 = (1./2.) * np.dot(np.einsum("ia,ia->i", vdto2, vdto2), masses)
		self.eta = etadto2 + (dt / (2.0 * self.Q)) * (kedto2 - (((3.0 * self.N + 1) / 2.0)) * self.kT)
		new_veloc = (vdto2 + (dt / 2.0) * new_accel) / (1.0 + (dt / 2.0) * self.eta)
		new_veloc[:6722] = np.zeros_like(new_veloc[:6722])
		return mols, coords, new_veloc, new_accel, potential, forces

class AndersenThermostatAlchem(Thermostat):
	def __init__(self, masses, init_veloc):
		"""
		Velocity Verlet step with a Langevin Thermostat
		"""
		self.m = masses.copy()
		self.N = len(masses)
		self.T = PARAMS["MDTemp"]  # Length of NH chain.
		self.gamma = 1.0 / 2.0 # Collision frequency (fs**-1)
		self.name = "Andersen"
		self.Rescale(init_veloc)
		print("Using ", self.name, " thermostat at ",self.T, " degrees Kelvin")
		return

	def step(self, force_field, mols, coords, veloc, accel, masses, delta, dt):
		coords = coords + veloc * dt + (1./2.) * accel * dt * dt
		for mol in mols:
			mol.coords = coords[:mol.NAtoms()]
		potential, forces = force_field(mols, delta)
		forces *= JOULEPERHARTREE
		new_accel = pow(10.0,-10.0) * np.einsum("ax,a->ax", forces, 1.0 / masses)
		new_veloc = veloc + (1./2.) * (accel + new_accel) * dt

		# Andersen velocity randomization.
		self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		s = np.sqrt(2.0 * self.gamma * self.kT / masses) # Mass is in kg,
		for i in range(coords.shape[0]):
			if (np.random.random() < self.gamma * dt):
				new_veloc[i] = np.random.normal(0.0, s[i], size = (3))
		return mols, coords, new_veloc, new_accel, potential, forces

def alchemical_transformation(force_field, mols, coords, transitions, name=None, cellsize=None):
	"""
	run an alchemical transformation molecular dynamics simulation

	Args:
		force_field: an energy-force routine
		mols (list): set of molecules in alignment
		transitions (list): list of lists of the first step and number of steps
							to perform a transformation
		PARAMS["MDMaxStep"]: Number of steps to take.
		PARAMS["MDTemp"]: Temperature to initialize or Thermostat to.
		PARAMS["MDdt"]: Timestep.
		PARAMS["MDV0"]: Sort of velocity initialization (None, or "Random")
		PARAMS["MDLogTrajectory"]: Write MD Trajectory.
	Returns:
		Nothing.
	"""
	if name is None:
		name = "alchem_traj"
	trajectory_set = MSet(name)
	maxsteps = PARAMS["MDMaxStep"]
	temp = PARAMS["MDTemp"]
	dt = PARAMS["MDdt"]
	initial_potential, initial_forces = force_field(mols, 0.0)
	potential = initial_potential
	EnergyStat = OnlineEstimator(initial_potential)
	traj_time = 0.0
	kinetic = 0.0
	num_atoms = max([mol.NAtoms() for mol in mols])
	atoms = np.zeros((len(mols), num_atoms), dtype=np.int32)
	masses = np.zeros((len(mols), num_atoms))
	for i, mol in enumerate(mols):
		atoms[i,:mol.NAtoms()] = mol.atoms
		masses[i,:mol.NAtoms()] = np.array(list(map(lambda x: ATOMICMASSES[x-1], atoms[i,:mol.NAtoms()])))
	md_log = None

	veloc = initialize_velocities(coords, masses, temp)
	veloc[:6722] = np.zeros_like(veloc[:6722])
	accel = np.zeros(coords.shape)
	thermostat = get_thermostat(masses[0], veloc)

	step = 0
	md_log = np.zeros((maxsteps, 7))
	write_trajectory(mols[0], name+"0")
	write_trajectory(mols[1], name+"1")
	while(step < maxsteps):
		t = time.time()
		traj_time = step*dt
		if step in range(transitions[0], transitions[0]+transitions[1]):
			delta = np.array(float((step - transitions[0] + 1.0) / transitions[1])).reshape((1))
		elif step > transitions[0]+transitions[1]:
			delta = np.array(1.0).reshape((1))
		else:
			delta = np.array(0.0).reshape((1))
		alchem_switch = np.where(np.not_equal(masses, 0), np.stack([np.tile(1.0 - delta, [masses.shape[1]]),
						np.tile(delta, [masses.shape[1]])]), np.zeros_like(masses))
		alchem_masses = np.sum(masses * alchem_switch, axis=0)
		if (PARAMS["MDThermostat"]==None):
			mols, coords, veloc, accel, potential, forces = velocity_verlet_step(force_field, mols, coords, veloc, accel, alchem_masses, delta, dt)
		else:
			mols, coords, veloc, accel, potential, forces = thermostat.step(force_field, mols, coords, veloc, accel, alchem_masses, delta, dt)
		if cellsize != None:
			coords = np.mod(coords, cellsize)
		kinetic = KineticEnergy(veloc, alchem_masses)
		md_log[step,0] = traj_time
		md_log[step,1] = delta
		md_log[step,4] = kinetic
		md_log[step,5] = potential
		md_log[step,6] = kinetic + (potential - initial_potential) * JOULEPERHARTREE
		avE, Evar = EnergyStat(potential) # I should log these.
		effective_temp = (2./3.) * kinetic / IDEALGASR

		if (step%3==0 and PARAMS["MDLogTrajectory"]):
			write_trajectory(mols[0], name+"0")
			write_trajectory(mols[1], name+"1")
		if (step%500==0):
			np.savetxt("./results/"+"MDLog"+name+".txt",md_log)

		step+=1
		LOGGER.info("%s Step: %i time: %.1f(fs) KE(kJ): %.5f PotE(Eh): %.5f ETot(kJ/mol): %.5f Teff(K): %.5f",
			name, step, traj_time, kinetic * len(alchem_masses) / 1000.0, potential,
			kinetic * len(alchem_masses) / 1000.0 + (potential) * KJPERHARTREE, effective_temp)
		print(("per step cost:", time.time() -t ))
	np.savetxt("./results/"+"MDLog"+name+".txt",md_log)
	return
