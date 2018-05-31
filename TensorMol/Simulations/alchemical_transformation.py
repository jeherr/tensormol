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

def write_trajectory(mol):
	m=Mol(atoms,self.x)
	m.properties["Time"]=time
	m.properties["KineticEnergy"]=kinetic
	m.properties["PotEnergy"]=potential
	m.WriteXYZfile("./results/", "MDTrajectory"+name)
	return

def initialize_velocities(coords, masses, temp):
	if PARAMS["MDV0"]=="Random":
		LOGGER.info("Using random initial velocities.")
		np.random.seed()
		velocity = np.random.randn(*coords[0].shape)
		# Do this elementwise otherwise H's blow off.
		for i in range(len(masses)):
			effective_temp = ((2.0 / (3.0 * IDEALGASR)) * pow(10.0,10.0) * (1./2.)
							* masses[0,i] * np.einsum("i,i", velocity[i], velocity[i]))
			print(effective_temp)
			if effective_temp != 0.0:
				velocity[i] *= np.sqrt(temp / effective_temp)
	elif PARAMS["MDV0"]=="Thermal":
		LOGGER.info("Using thermal initial velocities.")
		velocity = np.random.normal(size=coords.shape[0]) * np.sqrt(1.38064852e-23 * temp / masses)[:,None]
	return velocity

def get_thermostat(masses, velocity):
	if (PARAMS["MDThermostat"]=="Rescaling"):
		LOGGER.info("Using Rescaling thermostat.")
		thermostat = Thermostat(masses, velocity)
	elif (PARAMS["MDThermostat"]=="Nose"):
		LOGGER.info("Using Nose thermostat.")
		thermostat = NoseThermostat(masses, velocity)
	elif (PARAMS["MDThermostat"]=="Andersen"):
		LOGGER.info("Using Andersen thermostat.")
		thermostat = AndersenThermostat(masses, velocity)
	elif (PARAMS["MDThermostat"]=="Langevin"):
		LOGGER.info("Using Langevin thermostat.")
		thermostat = LangevinThermostat(masses, velocity)
	elif (PARAMS["MDThermostat"]=="NoseHooverChain"):
		LOGGER.info("Using NoseHooverChain thermostat.")
		thermostat = NoseChainThermostat(masses, velocity)
	else:
		LOGGER.info("No thermostat chosen. Performing unthermostatted MD.")
		thermostat = None
	return thermostat

def velocity_verlet_step(force_field, mols, delta, dt):
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
	coords = np.concatenate([mol.coords for mol in mols], axis=0)
	veloc = np.concatenate([mol.properties["velocity"] for mol in mols], axis=0)
	accel = np.concatenate([mol.properties["acceleration"] for mol in mols], axis=0)
	coords = coords + veloc * dt + (1./2.) * accel * dt * dt
	potential, forces = force_field(mols, delta)
	new_accel = pow(10.0,-10.0) * np.einsum("ax,a->ax", forces, 1.0 / masses)
	new_veloc = veloc + (1./2.) * (accel + new_accel) * dt
	new_mols = [mol.copy() for mol in mols]
	for mol in new_mols:
		mol.properties["velocity"] = new_veloc
		mol.properties["acceleration"] = new_accel
	return x,v,a,e

def alchemical_transformation(force_field, mols, transitions, name=None, cellsize=None):
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
	coords = np.zeros((len(mols), num_atoms, 3))
	for i, mol in enumerate(mols):
		atoms[i,:mol.NAtoms()] = mol.atoms
		coords[i,:mol.NAtoms()] = mol.coords
		masses[i,:mol.NAtoms()] = np.array(list(map(lambda x: ATOMICMASSES[x-1], atoms[i,:mol.NAtoms()])))
	md_log = None

	velocity = initialize_velocities(coords, masses, temp)
	acceleration = np.zeros(coords.shape)
	thermostat = get_thermostat(masses, velocity)

	step = 0
	md_log = np.zeros((maxsteps, 7))
	while(step < maxsteps):
		t = time.time()
		traj_time = step*dt
		if step == transitions[0]:
			for delta_step in range(transitions[1]):
				delta = np.array(float(delta_step / transitions[1])).reshape((1))
				if thermostat==None:
					coords, velocity, acceleration, potential = velocity_verlet_step(force_field, acceleration, mols, velocity, masses, dt)
				else:
					coords, velocity, acceleration, potential, forces = thermostat.step(force_field, acceleration, mols, velocity, masses, dt)
				if cellsize != None:
					coords = np.mod(coords, cellsize)
				kinetic = KineticEnergy(velocity, masses)
				md_log[step,0] = traj_time
				md_log[step,4] = kinetic
				md_log[step,5] = potential
				md_log[step,6] = kinetic + (potential - initial_potential) * JOULEPERHARTREE
				avE, Evar = EnergyStat(potential) # I should log these.
				effective_temp = (2./3.) * kinetic / IDEALGASR

				if (step%3==0 and PARAMS["MDLogTrajectory"]):
					WriteTrajectory()
				if (step%500==0):
					np.savetxt("./results/"+"MDLog"+name+".txt",md_log)

				step+=1
				LOGGER.info("%s Step: %i time: %.1f(fs) KE(kJ): %.5f PotE(Eh): %.5f ETot(kJ/mol): %.5f Teff(K): %.5f", name, step, traj_time, kinetic * len(self.m) / 1000.0, potential, kinetic * len(self.m) / 1000.0 + (potential) * KJPERHARTREE, effective_temp)
				#LOGGER.info("Step: %i time: %.1f(fs) <KE>(kJ/mol): %.5f <|a|>(m/s2): %.5f <EPot>(Eh): %.5f <Etot>(kJ/mol): %.5f Teff(K): %.5f", step, self.t, self.KE/1000.0,  np.linalg.norm(self.a) , self.EPot, self.KE/1000.0+self.EPot*KJPERHARTREE, Teff)
				print(("per step cost:", time.time() -t ))
		#self.KE = KineticEnergy(self.v,self.m)
		#Teff = (2./3.)*self.KE/IDEALGASR
		if (PARAMS["MDThermostat"]==None):
			coords , velocity, acceleration, potential = VelocityVerletStep(force_field, acceleration, mols, velocity, masses, dt)
		else:
			coords , velocity, acceleration, potential, forces = thermostat.step(force_field, acceleration, mols, velocity, masses, dt)
		if cellsize != None:
			coords  = np.mod(self.x, cellsize)
		kinetic = KineticEnergy(velocity, masses)
		md_log[step,0] = traj_time
		md_log[step,4] = kinetic
		md_log[step,5] = potential
		md_log[step,6] = kinetic + (potential - initial_potential) * JOULEPERHARTREE
		avE, Evar = EnergyStat(potential) # I should log these.
		effective_temp = (2./3.) * kinetic / IDEALGASR

		if (step%3==0 and PARAMS["MDLogTrajectory"]):
			WriteTrajectory()
		if (step%500==0):
			np.savetxt("./results/"+"MDLog"+name+".txt",md_log)

		step+=1
		LOGGER.info("%s Step: %i time: %.1f(fs) KE(kJ): %.5f PotE(Eh): %.5f ETot(kJ/mol): %.5f Teff(K): %.5f", name, step, traj_time, kinetic * len(self.m) / 1000.0, potential, kinetic * len(self.m) / 1000.0 + (potential) * KJPERHARTREE, effective_temp)
		#LOGGER.info("Step: %i time: %.1f(fs) <KE>(kJ/mol): %.5f <|a|>(m/s2): %.5f <EPot>(Eh): %.5f <Etot>(kJ/mol): %.5f Teff(K): %.5f", step, self.t, self.KE/1000.0,  np.linalg.norm(self.a) , self.EPot, self.KE/1000.0+self.EPot*KJPERHARTREE, Teff)
		print(("per step cost:", time.time() -t ))
	np.savetxt("./results/"+"MDLog"+name+".txt",md_log)
	return
