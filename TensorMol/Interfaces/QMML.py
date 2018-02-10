"""
Routines for running external QMML Calculations
"""
from __future__ import absolute_import
from __future__ import print_function
from ..TMParams import *
from ..Containers import *
from ..TFNetworks import *
from .AbInitio import *

def QMMLEnergy(manager_, m_, qm_atom_count_, method_="dft", basis_="631+g*", functional_="b3lyp"):
	atoms = m_.atoms
	atom_coords = m_.coords
	a=MSet()
	a.mols.append(m_)

	def CalculateProperties(molecule):
		Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager_.EvalBPDirectEEUpdateSingle(molecule, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
		energy         = Etotal[0]
		force          = gradient[0]
		partial_charge = atom_charge[0]
		dipole         = mol_dipole[0]
		return {"energy": energy, "partial_charge": partial_charge, "dipole": dipole, "force": force}

	def CalculateEnergy(atomic_coords):
		full_molecule = Mol()
		qm_molecule   = Mol()

		full_molecule.atoms  = atoms
		full_molecule.coords = atom_coords
		qm_molecule.atoms    = atoms[:qm_atom_count_]
		qm_molecule.coords   = atom_coords[:qm_atom_count_]
		qm_atomic_coords     = []
		i = 0
		while i < len(qm_molecule.atoms):
			qm_atomic_coords.append(["{}".format(qm_molecule.atoms[i]), (qm_molecule.coords[i, 0], qm_molecule.coords[i, 1], qm_molecule.coords[i, 2])])
			i = i + 1

		# -----------------------
		#     ML
		# -----------------------
		properties     = CalculateProperties(full_molecule)
		ml_full_energy = properties["energy"]
		ml_full_force  = properties["force"]/2625499.638
		ml_full_partial_charge = properties["partial_charge"]

		properties         = CalculateProperties(qm_molecule)
		ml_qmregion_energy = properties["energy"]
		ml_qmregion_force  = properties["force"]/2625499.638

		# -----------------------
		#     QM
		# -----------------------
		qm_energy = 0
		if method_ is "dft":
			qm_energy = PyscfDft(m_, basis_, functional_)
		elif method_ is "ccsd":
			qm_energy = PyscfCcsd(m_, basis_)
		elif method_ is "ccsd_t":
			qm_energy = PyscfCcsdt(m_, basis_)
		else:
			LOGGER.error("QM Method not found for QM-ML")

		# gradients = my_dft.apply(grad.RKS).grad()
		# qm_force = gradients

		# -----------------------
		#     QM-ML
		# -----------------------
		qmml_energy = ml_full_energy + qm_energy - ml_qmregion_energy
		# qmml_force  = np.copy(ml_full_force)
		# qmml_force[:qm_atom_count_] = qmml_force[:qm_atom_count_] + qm_force - ml_qmregion_force
		# print("QM-ML Force:")
		# print(qmml_force)
		# print("TM-Full Force:")
		# print(ml_full_force)
		return qmml_energy

	return CalculateEnergy(m_)
