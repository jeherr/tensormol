"""
This Routine Requires PYSCF at Minimum
"""
import os
import subprocess
from pyscf import gto, dft, grad

str0 = "$molecule\n0 1\n"
str1 = "$end\n\n$rem\njobtype force"
str2 = "\nbasis 6-311g**\nexchange omegaB97X-D\nthresh 11\nsymmetry false\nsym_ignore true\nunrestricted true\nGEOM_OPT_MAX_CYCLES 2000\nMAX_SCF_CYCLES 400$end"

MAX_ATOMIC_NUMBER = 55

class MLMM:
	def __init__(self,nn,molAB,Nenv, mm = "OB")):
		"""
		Assuming John's OpenBabel routine works
		"""
		self.NN = nn
		self.molAB = molAB
		self.molA = self.SeparateA(molAB,Nenv)
		return

	def SeparateA(self,mol,a):
                """
                Make Mole A from Whole Molecule set knowing where the environment starts
                """
		self.na = a
		self.nb = len(mol.atoms) - a
		self.natoms = len(mol.atoms)
		geom = str(self.na) + "\n"
		for i in range(self.na):
			geom += "\n" + str(AtomicSymbol(int(mol.atoms[i]))) + " " + str(mol.coords[i,0]) + " " + str(mol.coords[i,1]) + " " + str(mol.coords[i,2])
		m = Mol()
		m.FromXYZString(geom)
		return m

	def GetEnergyForceRoutine(self, m):
		self.MLA = self.NN.GetEnergyForceRoutine(self.molA)
		def EF(xyz_,DoForce = True, Debug = False):
			# MM
			e, f = ob_singlepoint(self.molAB, forcefield = "MMFF94", forces = True)
			# MMA
			eA, fA = ob_singlepoint(self.molA, forcefield = "MMFF94", forces = True)
			# MLA
			xyza = xyz_[:self.na,:]
			eAml, fAml = self.MLA(xyza,True, Debug)
			# energy
			finE = e - eA + eAml
			# forces
			finF = f.copy()
			finF[:self.na,:] += - fA + fAml
			return finE, finF
		return EF

class QMML:
	"""
	Energy and Force routine
	Combining QM and ML
	"""
	def __init__(self,nn,molAB,Nenv, qm = "PYSCF"):
		"""
		nn:
			network (default basis = 6-311g**, xc = omegaB97X-D)
		qm:
			QM routine that returns Energy and force
			input: 
		molA:
			TM molecule variable in area A
		molB:
			TM molecule variable in area B
					
		"""
		self.NN = nn
		self.qm = qm.upper()
		self.molAB = molAB
		if self.qm == "PYSCF" or self.qm == "QCHEM":
			print("QM Method: ", self.qm)
			self.QM = self.QMroutine(self.qm)
		else:
			print("QM Method not specified")
			quit()			
		self.molA = self.SeparateA(molAB,Nenv)
		
		return


	def combineMol(self,molA,molB):
		self.na = len(molA.atoms)
		self.nb = len(molB.atoms)
		self.natoms = self.na + self.nb
		atoms = np.append(molA.atoms,molB.atoms)
		coords = np.concatenate((molA.coords,molB.coords))
		comb = np.column_stack((atoms,coords))
		geom = str(self.natoms) + "\n"
		for i in range(self.natoms):
			geom += "\n" + str(AtomicSymbol(int(comb[i,0]))) + " " + str(comb[i,1]) + " " + str(comb[i,2]) + " " + str(comb[i,3])
		m = Mol()
		m.FromXYZString(geom)
		return m

	def SeparateA(self,mol,a):
		"""
		Make Mole A from Whole Molecule set knowing where the environment starts
		"""
		self.na = a
		self.nb = len(mol.atoms) - a
		self.natoms = len(mol.atoms)
		geom = str(self.na) + "\n"
		for i in range(self.na):
			geom += "\n" + str(AtomicSymbol(int(mol.atoms[i]))) + " " + str(mol.coords[i,0]) + " " + str(mol.coords[i,1]) + " " + str(mol.coords[i,2])
		m = Mol()
		m.FromXYZString(geom)
		return m
		

	def QMroutine(self,qm):
		if self.qm == "PYSCF":
			def EF(molstr,basis = "6-311g**", xc = "PBE0"):
				m1 = gto.Mole()
				m1.atom = molstr
				m1.basis = basis
				m1.build()
				mf = dft.RKS(m1)
				mf.xc = xc
				mf.kernel()
				energy = mf.e_tot
				force = grad.RKS(mf).kernel()/0.52917721092 * -JOULEPERHARTREE
				return energy, force
		elif self.qm == "QCHEM":
			def EF(molstr,basis = "6-311g**", xc = "omegaB97X-D", nt = 64):
				with open("./tmp.in","w") as fin:
					input = str0+molstr+str1+str2
					fin.write(input)
				subprocess.run("qchem -nt " + str(nt) +  " tmp.in tmp.out", shell=True, check = True)
				natom = molstr.count("\n")
				energy = None
				j = float("inf")
				k = 0
				l = 0
				force = np.zeros((int(natom/6+1)*3,6))
				with open("./tmp.out","r") as fin:
					for num,line in enumerate(fin):
						if "Total energy in the final basis set =" in line:
							energy = line.split()
							energy = np.array(energy)[-1]
						if "Gradient of SCF Energy" in line:
							j = num + 1
						if num >= j and num < int(j) + int(natom/6+1)*4:
							if int(k%4) != 0:
								force[l,:] += np.resize(np.array(line.split()[1:]).astype(float),(6))
								l += 1
							k += 1
				force[int(natom/6)*3:,int(natom%6):] *= 0
				nrow = int(natom/6+1)*6
				temp = np.resize(force.T,(nrow,3))
				rr = int(natom/6+1)
				ind = np.arange(0,nrow,rr)
				for ii in range(1,rr):
					ind = np.append(ind,np.arange(ii,nrow,rr))
				force = temp[ind,:]
				force = force[:natom,:]/0.52917721092 * -JOULEPERHARTREE
				return energy, force
		return EF

	def GetEnergyForceRoutine(self, m):
		self.ML = self.NN.GetEnergyForceRoutine(self.molAB)
		self.MLA = self.NN.GetEnergyForceRoutine(self.molA)

		def EF(xyz_,DoForce = True, Debug = False):
			# ML AB
			e, f = self.ML(xyz_,True, Debug)
			# ML A
			xyza = xyz_[:self.na,:]
			eA, fA = self.MLA(xyza,True, Debug)
			# QM
			comb = np.column_stack((self.molA.atoms,xyza))
			molstrA = ""
			n = len(self.molA.atoms)
			for i in range(n):
				molstrA += str(int(comb[i,0])) + " " + str(comb[i,1]) + " " + str(comb[i,2]) + " " + str(comb[i,3])+ "\n"
			eAqc, fAqc = self.QM(molstrA)
			
			finE = float(e) - float(eA) + float(eAqc)
			finF = f.copy()
			finF[:self.na,:] += - fA + fAqc
			return finE, finF
		return EF
		
