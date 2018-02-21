"""
Routines for running calculations from OpenBabel
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
# import openbabel as ob
# import pybel as pb
from ..Util import *
import random, math, subprocess

def ob_singlepoint(mol, forcefield="MMFF94", forces=True):
	obconv = ob.OBConversion()
	obconv.SetInFormat("xyz")
	obmol = ob.OBMol()
	mol.WriteXYZfile(fname="obtmp", mode="w")
	obconv.ReadFile(obmol,"./obtmp.xyz")
	ob_forces = np.zeros((obmol.NumAtoms(), 3))
	ff = ob.OBForceField.FindType(forcefield)
	ff.Setup(obmol)
	ff.GetCoordinates(obmol)
	data = ob.toConformerData(obmol.GetData(4))
	frc=data.GetForces()
	for i in range(obmol.NumAtoms()):
		ob_forces[i,0] = frc[0][i].GetX()
		ob_forces[i,1] = frc[0][i].GetY()
		ob_forces[i,2] = frc[0][i].GetZ()
	if forces:
		return ff.Energy(), ob_forces
	else:
		return ff.Energy()

def ob_minimize_geom(mol, forcefield="MMFF94"):
	obconv = ob.OBConversion()
	obconv.SetInAndOutFormats("xyz", "xyz")
	obmol = ob.OBMol()
	mol.WriteXYZfile(fname="obtmp", mode="w")
	obconv.ReadFile(obmol,"./obtmp.xyz")
	ff = ob.OBForceField.FindForceField(forcefield)
	if ff == 0:
		print("Could not find forcefield")
	ff.SetLogLevel(ob.OBFF_LOGLVL_LOW)
	ff.SetLogToStdErr()
	if ff.Setup(obmol) == 0:
		print("Could not setup forcefield")
	ff.ConjugateGradients(2000)
	ff.GetCoordinates(obmol)
	pbmol = pb.Molecule(obmol)
	min_coords = np.zeros((len(pbmol.atoms),3))
	for i, atom in enumerate(pbmol.atoms):
		min_coords[i] = atom.coords
	return Mol(mol.atoms, min_coords)
