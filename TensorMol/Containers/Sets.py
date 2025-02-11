#
# A molecule set is not a training set.
#
from __future__ import absolute_import
from __future__ import print_function
from .Mol import *
from ..Util import *
#from .MolGraph import *
#from .MolFrag import *
import numpy as np
import os,sys,re,copy,time
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle

class MSet:
	""" A molecular database which
		provides structures """
	def __init__(self, name_ ="gdb9", path_="./datasets/", center_=True):
		self.mols=[]
		self.path=path_
		self.name=name_
		self.suffix=".pdb" #Pickle Database? Poor choice. | We should rename to .mset or .mst - jeherr
		self.center=center_

	def Save(self, filename=None):
		for mol in self.mols:
			mol.Clean()
		if filename == None:
			filename = self.name
		elif filename != None:
			self.name = filename
		LOGGER.info("Saving set to: %s ", self.path+filename+self.suffix)
		f=open(self.path+filename+self.suffix,"wb")
		if sys.version_info[0] < 3:
			pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			pickle.dump(self.__dict__, f)
		f.close()
		return

	def Load(self, filename=None):
		if filename == None:
			filename = self.name
		from ..Containers.PickleTM import UnPickleTM as UnPickleTM
		print("Loading Set: ", self.path+filename+self.suffix)
		tmp = UnPickleTM(self.path+filename+self.suffix)
		self.__dict__.update(tmp)
		LOGGER.info("Loaded, "+str(len(self.mols))+" molecules "+str(self.NAtoms())+" Atoms total "+str(self.AtomTypes())+" Types ")
		return

	def RemoveElementAverages(self):
		"""
		Removes average from energies and charges returns dictionaries mapping
		AN=> element averages useful for normalizing training.
		It does this using least-sq. This eliminates the need for element
		atomization energies.

		Returns:
			AvE,AvQ: dictionaries mapping Atomic number onto the averages.
		"""
		if (not "energy" in self.mols[-1].properties.keys()):
			return {},{}
		At = self.AtomTypes().tolist()
		AvE = {x:0. for x in At}
		AvQ = {x:0. for x in At}
		nmols = len(self.mols)
		nele = len(AvE)
		neled = max(At)+1
		b = np.zeros(nmols)
		# Use dense zero-padded arrays to avoid index logic.
		a = np.zeros((nmols,neled))
		noa = np.zeros(neled)
		coa = np.zeros(neled)
		for i,m in enumerate(self.mols):
			unique, counts = np.unique(m.atoms, return_counts=True)
			stoich = np.zeros(neled)
			for j in range(len(unique)):
				stoich[unique[j]] = counts[j]
			a[i] += stoich
			noa += stoich
			b[i] = m.properties["energy"]
			for atom in range(m.NAtoms()):
				try:
					coa[m.atoms[atom]]+=m.properties["charges"][atom]
				except:
					coa[m.atoms[atom]]+=m.properties["mul_charge"][atom]
		x,r = np.linalg.lstsq(a,b)[:2]
		averageqs = coa/noa
		for e in At:
			AvE[e] = x[e]
			AvQ[e] = averageqs[e]
		# Report the residual information.
		EErrors = np.zeros(nmols)
		QErrors = np.zeros(nmols)
		for i,m in enumerate(self.mols):
			e0 = 0.0
			for j,e in enumerate(At):
				e0 += AvE[e]*a[i,e]
			#print("Formula: ",a[i],m.properties["energy"],e0)
			EErrors[i] = e0 - m.properties["energy"]
		print("---- Results of Stoichiometric Model ----")
		print("MAE  Energy: ", np.average(np.abs(EErrors)))
		print("MXE  Energy: ", np.max(np.abs(EErrors)))
		print("RMSE Energy: ", np.sqrt(np.average(EErrors*EErrors)))
		print("AvE: ", AvE)
		print("AvQ: ", AvQ)
		self.AvE = AvE
		self.AvQ = AvQ
		return AvE,AvQ

	def cut_max_atomic_number(self,max_an):
		cut_down_mols = []
		for mol in self.mols:
			if (np.max(mol.atoms) < max_an):
				cut_down_mols.append(mol)
		self.mols = cut_down_mols

	def keep_atomic_numbers(self, atomic_nums):
		new_mols = []
		atomic_nums = np.array(atomic_nums, dtype=np.uint8)
		for mol in self.mols:
			if np.all(np.isin(mol.atoms, atomic_nums)):
				new_mols.append(mol)
		self.mols = new_mols

	def cut_unique_bond_hash(self):
		cut_down_mols = []
		known_hashes = []
		for mol in self.mols:
			if (not mol.properties['bond_hash'] in known_hashes):
				cut_down_mols.append(mol)
				known_hashes.append(mol.properties['bond_hash'])
		self.mols = cut_down_mols

	def cut_min_num_atoms(self, min_n_atoms):
		cut_down_mols = []
		for mol in self.mols:
			if mol.atoms.shape[0] >= min_n_atoms:
				cut_down_mols.append(mol)
		self.mols = cut_down_mols

	def cut_max_num_atoms(self, max_n_atoms):
		cut_down_mols = []
		for mol in self.mols:
			if mol.atoms.shape[0] <= max_n_atoms:
				cut_down_mols.append(mol)
		self.mols = cut_down_mols

	def cut_randomselection(self, n_totake=100000.):
		accept_fraction = (n_totake/(len(self.mols)))
		cut_down_mols = []
		for mol in self.mols:
			if (random.random()<accept_fraction):
				cut_down_mols.append(mol)
		self.mols = cut_down_mols

	def cut_max_grad(self, max_grad=1.0):
		cut_down_mols = []
		for mol in self.mols:
			if (np.max(np.abs(mol.properties['gradients']))<max_grad):
				cut_down_mols.append(mol)
		self.mols = cut_down_mols

	def cut_energy_outliers(self,max_diff=1.0):
		"""
		removes any molecules which are more than a hartree away from the mean.
		"""
		self.RemoveElementAverages()
		cut_down_mols = []
		for m in self.mols:
			unique, counts = np.unique(m.atoms, return_counts=True)
			e0=0.
			for i in range(len(unique)):
				e0+=counts[i]*self.AvE[unique[i]]
			if abs(m.properties["energy"] - e0) < max_diff:
				cut_down_mols.append(m)
		self.mols = cut_down_mols
		return

	def DistortAlongNormals(self, npts=8, random=True, disp=.2):
		'''
		Create a distorted copy of a set

		Args:
			npts: the number of points to sample along the normal mode coordinate.
			random: whether to randomize the order of the new set.
			disp: the maximum displacement of atoms along the mode

		Returns:
			A set containing distorted versions of the original set.
		'''
		print("Making distorted clone of:", self.name)
		s = MSet(self.name+"_NEQ")
		ord = range(len(self.mols))
		if(random):
			np.random.seed(int(time.time()))
			ord=np.random.permutation(len(self.mols))
		for j in ord:
			newcoords = self.mols[j].ScanNormalModes(npts,disp)
			for i in range(newcoords.shape[0]): # Loop modes
				for k in range(newcoords.shape[1]): # loop points
					s.mols.append(Mol(self.mols[j].atoms,newcoords[i,k,:,:]))
					s.mols[-1].DistMatrix = self.mols[j].DistMatrix
		return s

	def RotatedClone(self, NRots=3):
		"""
		Rotate every molecule NRots Times.
		We should toss some reflections in the mix too...
		"""
		print("Making Rotated clone of:", self.name)
		s = MSet(self.name)
		ord = range(len(self.mols))
		if(random):
			np.random.seed(int(time.time()))
			ord=np.random.permutation(len(self.mols))
		for j in ord:
			for i in range (0, NRots):
				s.mols.append(copy.deepcopy(self.mols[j]))
				s.mols[-1].coords -= s.mols[-1].Center()
				s.mols[-1].RotateRandomUniform()
		return s

	def DistortedClone(self, NDistorts=1, random=True):
			''' Create a distorted copy of a set'''
			print("Making distorted clone of:", self.name)
			s = MSet(self.name+"_NEQ")
			ord = range(len(self.mols))
			if(random):
				np.random.seed(int(time.time()))
				ord=np.random.permutation(len(self.mols))
			for j in ord:
				for i in range (0, NDistorts):
					s.mols.append(copy.deepcopy(self.mols[j]))
					s.mols[-1].Distort()
			return s

	def TransformedClone(self, transfs):
		''' make a linearly transformed copy of a set. '''
		LOGGER.info("Making Transformed clone of:"+self.name)
		s = MSet(self.name)
		ord = range(len(self.mols))
		for j in ord:
			for k in range(len(transfs)):
				s.mols.append(copy.deepcopy(self.mols[j]))
				s.mols[-1].Transform(transfs[k])
		return s

	def CenterSet(self):
		"""
		Translates every Mol such that the center is at 0.
		"""
		for mol in self.mols:
			mol.coords -= mol.Center()

	def NAtoms(self):
		nat=0
		for m in self.mols:
			nat += m.NAtoms()
		return nat

	def MaxNAtom(self):
		if (len(self.mols)>0):
			return np.max([m.NAtoms() for m in self.mols])
		else:
			return 0

	def max_neighbors(self):
		return max([mol.max_neighbors() for mol in self.mols])

	def AtomTypes(self):
		types = np.array([],dtype=np.uint8)
		for m in self.mols:
			types = np.union1d(types,m.AtomTypes())
		return types

	def element_counts(self):
		atoms = [mol.atoms for mol in self.mols]
		atoms = np.concatenate(atoms)
		elements, counts = np.unique(atoms, return_counts=True)
		print("Elements present in set: ", elements)
		print("Amount of each element: ", counts)

	def max_atomic_num(self):
		types = np.array([],dtype=np.uint8)
		for m in self.mols:
			types = np.union1d(types,m.AtomTypes())
		return np.max(types)

	def BondTypes(self):
		return np.asarray([x for x in itertools.product(self.AtomTypes().tolist(), repeat=2)])

	def read_xyz_set_with_properties(self, path, properties=[]):
		"""
		Reads xyz files from a directory with properties in the comment line and adds them to the
		set.mols list

		Args:
			path (string): The location of the xyz files to be read
			properties (list of strings): A list of properties in the comment line of the file.

		Notes:
			Properties in the xyz file must be semi-colon delimited and in the same order as specified in
			the properties variable.

			Components of properties must be comma delimited and are expected in the same order as the atoms
			with ordering of x, y, z if necessary.

			Currently only valid properties are "name", "energy", "forces", "dipole", and "mulliken_charges".
			Other properties to be added as needed.
		"""
		from .Mol import Mol
		from os import listdir
		from os.path import isfile, join
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		for file in onlyfiles:
			if ( file[-4:]!='.xyz' ):
					continue
			self.mols.append(Mol())
			self.mols[-1].read_xyz_with_properties(path+file, properties, self.center)
		return

	def ReadXYZUnpacked(self, path="/Users/johnparkhill/gdb9/", has_energy=False, has_force=False, has_charge=False, has_mmff94=False):
		"""
		Reads XYZs in distinct files in one directory as a molset
		Args:
			path: the directory which contains the .xyz files to be read
			has_energy: switch to turn on reading the energy from the comment line as formatted from the md_dataset on quantum-machine.org
			has_force: switch to turn on reading the force from the comment line as formatted from the md_dataset on quantum-machine.org
		"""
		from os import listdir
		from os.path import isfile, join
		#onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		for file in onlyfiles:
			if ( file[-4:]!='.xyz' ):
					continue
			self.mols.append(Mol())
			self.mols[-1].ReadGDB9(path+file, file)
			self.mols[-1].properties["set_name"] = self.name
			if has_force:
				self.mols[-1].ForceFromXYZ(path+file)
			if has_energy:
				self.mols[-1].EnergyFromXYZ(path+file)
			if has_charge:
				self.mols[-1].ChargeFromXYZ(path+file)
			if has_mmff94:
				self.mols[-1].MMFF94FromXYZ(path+file)
		if (self.center):
			self.CenterSet()
		return

	def ReadXYZ(self,filename = None, xyz_type = 'mol'):
		"""
		Reads XYZs concatenated into a single file separated by \n\n as a molset
		"""
		from .Mol import Mol
		if filename == None:
			filename = self.name
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].strip() and
					all([x.isdigit() for x in txts[line].split()])):
				line0=line
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(Mol())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]))
				self.mols[-1].name = str(txts[line0+1])
				self.mols[-1].properties["set_name"] = self.name
		if (self.center):
			self.CenterSet()
		LOGGER.debug("Read "+str(len(self.mols))+" molecules from XYZ")
		return

	def AppendFromDirectory(self, apath_):
		"""
		Append all xyz files in apath_ to this set.
		"""
		for file in os.listdir(apath_):
			if file.endswith(".xyz"):
				m = Mol()
				m.properties = {"from_file":file}
				f = open(file,'r')
				fs = f.read()
				m.FromXYZString(fs)
				self.mols.append(m)
		return

	def WriteXYZ(self,filename=None):
		if filename == None:
			filename = self.name
		for mol in self.mols:
			mol.WriteXYZfile(self.path,filename)
		LOGGER.info('Wrote %s ', filename)
		return

	def pop(self, ntopop):
		for i in range(ntopop):
			self.mols.pop()
		return

	def OnlyWithElements(self, allowed_eles):
		"""
		Removes molecules with unwanted atoms from a set.
		"""
		mols=[]
		for mol in self.mols:
			if set(list(mol.atoms)).issubset(allowed_eles):
				mols.append(mol)
		for i in allowed_eles:
			self.name += "_"+str(i)
		self.mols=mols
		return

	def OnlyAtoms(self,allowed_eles):
		"""
		Removes any unwanted atoms from the set.
		"""
		for mol in self.mols:
			included = []
			for i,atom in enumerate(mol.atoms):
				if (atom in allowed_eles):
					included.append(i)
			mol.atoms = mol.atoms[included]
			mol.coords = mol.coords[included]
		return

	def AppendSet(self, b):
		if (self.name == None):
			self.name = self.name + b.name
		self.mols = self.mols+b.mols
		return

	def rms(self,other_):
		if (len(self.mols) != len(other_.mols)):
			raise Exception("Bad Comparison")
		rmss = [self.mols[i].rms_inv(other_.mols[i]) for i in range(len(self.mols))]
		return rmss

	def Statistics(self):
		""" Return some energy information about the samples we have... """
		print("Set Statistics----")
		print("Nmol: ", len(self.mols))
		natoms = [m.NAtoms() for m in self.mols]
		print("Ave Num of atoms: ", np.mean(natoms))
		print("Max Num of atoms: ", np.max(natoms), " mol num: ",natoms.index(np.max(natoms)))
		print(self.mols[natoms.index(np.max(natoms))])
		print("Min Num of atoms: ", np.min(natoms), " mol num: ",natoms.index(np.min(natoms)))
		print(self.mols[natoms.index(np.min(natoms))])
		print(np.histogram(natoms))
		if 'energy' in self.mols[0].properties:
			energies = [m.properties['energy'] for m in self.mols]
			print("Average Energy: ",np.mean(energies))
		if 'gradients' in self.mols[0].properties:
			avgrad = [np.mean(m.properties['gradients'],axis=0) for m in self.mols]
			avgradnorm = [np.mean(np.linalg.norm(m.properties['gradients'],axis=-1)) for m in self.mols]
			print('AverageGrad',np.mean(avgrad))
			print('AverageGradNorm',np.mean(avgradnorm))
			print('StdGradNorm',np.std(avgradnorm))
			print('MaxGradNorm',np.max(avgradnorm))
		if 'charges' in self.mols[0].properties:
			charges = [np.mean(m.properties['charges'],axis=0) for m in self.mols]
			abscharges = [np.max(np.abs(m.properties['charges']),axis=0) for m in self.mols]
			print('AverageCharge',np.mean(charges))
			print('AverageMaxAbsCharge',np.mean(abscharges))
			print('StdMaxAbsCharge',np.std(abscharges))
			print('MaxMaxAbsCharge',np.max(abscharges))
		return

	def WriteSmiles(self):
		for mol in self.mols:
			mol.WriteSmiles()
		return

	def MakeBonds(self):
		self.NBonds = 0
		for m in self.mols:
			self.NBonds += m.MakeBonds()
		self.BondTypes = np.unique(np.concatenate([m.bondtypes for m in self.mols],axis=0),axis=0)

class FragableMSet(MSet):
	def __init__(self, name_ ="NaClH2O", path_="./datasets/"):
		MSet.__init__(self, name_, path_)
		return

	def ReadGDB9Unpacked(self, path="/Users/johnparkhill/gdb9/"):
		""" Reads the GDB9 dataset as a pickled list of molecules"""
		from os import listdir
		from os.path import isfile, join
		#onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		for file in onlyfiles:
			if ( file[-4:]!='.xyz' ):
				continue
			self.mols.append(FragableCluster())
			self.mols[-1].ReadGDB9(path+file, file)
			self.mols[-1].properties["set_name"] = self.name
			self
		return

	def ReadXYZ(self,filename, xyz_type = 'mol'):
		""" Reads XYZs concatenated into a single separated by \n\n file as a molset """
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].count('Comment:')>0):
				line0=line-1
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(FragableCluster())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]))
				self.mols[-1].name = str(line)
				self.mols[-1].properties["set_name"] = self.name
		return

	def MBE(self,  atom_group=1, cutoff=10, center_atom=0):
		for mol in self.mols:
			mol.MBE(atom_group, cutoff, center_atom)
		return

	def PySCF_Energy(self):
		for mol in self.mols:
			mol.properties["energy"] = PySCFMP2Energy(mol)
		return

	def Generate_All_MBE_term(self,  atom_group=1, cutoff=10, center_atom=0, max_case = 1000000):
		for mol in self.mols:
			mol.Generate_All_MBE_term(atom_group, cutoff, center_atom, max_case)
		return

	def Generate_All_MBE_term_General(self, frag_list=[], cutoff=10, center_atom=0):
		for mol in self.mols:
			mol.Generate_All_MBE_term_General(frag_list, cutoff, center_atom)
		return

	def Calculate_All_Frag_Energy(self, method="pyscf"):
		for mol in self.mols:
			mol.Calculate_All_Frag_Energy(method)
			# 	mol.Set_MBE_Energy()
		return

	def Calculate_All_Frag_Energy_General(self, method="pyscf"):
		for mol in self.mols:
			#print mol.properties
			#print "Mol set_name", mol.properties["set_name"]
			mol.Calculate_All_Frag_Energy_General(method)
			#        mol.Set_MBE_Energy()
		return

	def Get_All_Qchem_Frag_Energy(self):
		for mol in self.mols:
			mol.Get_All_Qchem_Frag_Energy()
		return

	def Get_All_Qchem_Frag_Energy_General(self):
		for mol in self.mols:
			mol.Get_All_Qchem_Frag_Energy_General()
		return

	def Generate_All_Pairs(self, pair_list=[]):
		for mol in self.mols:
			mol.Generate_All_Pairs(pair_list)
		return

	def Get_Permute_Frags(self, indis=[0]):
		for mol in self.mols:
			mol.Get_Permute_Frags(indis)
		return

class FragableMSetBF(FragableMSet):
	def __init__(self, name_ ="NaClH2O", path_="./datasets/"):
		MSet.__init__(self, name_, path_)
		return

	def ReadXYZ(self,filename, xyz_type = 'mol'):
		""" Reads XYZs concatenated into a single separated by \n\n file as a molset """
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].count('Comment:')>0):
				line0=line-1
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(FragableClusterBF())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]))
				self.mols[-1].name = str(line)
				self.mols[-1].properties["set_name"] = self.name
		return

	def Generate_All_MBE_term_General(self, frag_list=[]):
		for mol in self.mols:
			mol.Generate_All_MBE_term_General(frag_list)
		return




class GraphSet(MSet):
	def __init__(self, name_ ="gdb9", path_="./datasets/", center_=True):
		MSet.__init__(self, name_, path_, center_)
		self.graphs=[]
		self.path=path_
		self.name=name_
		self.suffix=".graph" #Pickle Database? Poor choice.

	def BondTypes(self):
		types = np.array([],dtype=np.uint8)
		for m in self.mols:
			types = np.union1d(types,m.BondTypes())
		return types

	def NBonds(self):
		nbonds=0
		for m in self.mols:
			nbonds += m.NBonds()
		return nbonds

	def MakeGraphs(self):
		graphs = map(MolGraph, self.mols)
		return graphs

	def ReadXYZ(self,filename = None, xyz_type = 'mol', eqforce=False):
		""" Reads XYZs concatenated into a single file separated by \n\n as a molset """
		if filename == None:
			filename = self.name
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].count('Comment:')>0):
				line0=line-1
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(MolGraph())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]))
				self.mols[-1].name = str(txts[line0+1])
				self.mols[-1].properties["set_name"] = self.name
		if (self.center):
			self.CenterSet()
		if (eqforce):
			self.EQ_forces()
		LOGGER.debug("Read "+str(len(self.mols))+" molecules from XYZ")
		return

	def Save(self):
		print("Saving set to: ", self.path+self.name+self.suffix)
		f=open(self.path+self.name+self.suffix,"wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
		return

	def Load(self):
		f = open(self.path+self.name+self.suffix,"rb")
		tmp=pickle.load(f)
		self.__dict__.update(tmp)
		f.close()
		print("Loaded, ", len(self.mols), " molecules ")
		print(self.NAtoms(), " Atoms total")
		print(self.AtomTypes(), " Types ")
		return
