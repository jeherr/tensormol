import math as m
import numpy as np
import copy
from ..Containers.Sets import *

class ZmatTools:
	'''A coordinate converter class'''
	def __init__(self):
		# Dictionary of the masses of elements indexed by element name;
		# includes X for dummy atoms
		self.cartesian = []
		self.zmatrix = []

	def DihedralScans(self,mol):
		tore = MSet("Dihedrals")
		natom = mol.NAtoms()
		tore.mols.append(mol)
		self.read_cartesian(mol.atoms, mol.coords)
		self.cartesian_to_zmatrix()
		zmat0 = copy.deepcopy(self.zmatrix)
		for j in range(3,natom):
			for angle in np.linspace(-Pi,Pi,10):
				self.zmatrix = copy.deepcopy(zmat0)
				atoma = self.zmatrix[j][1][1][0]
				atomb = self.zmatrix[j][1][2][0]
				self.zmatrix[j][1][2][1] = angle
				print(angle,self.zmatrix)
				self.zmatrix_to_cartesian()
				from ..Containers.Mol import Mol
				atoms = np.array([c[0] for c in self.cartesian],dtype=np.int32)
				coords = np.zeros((natom,3))
				for k in range(natom):
					coords[k] = self.cartesian[k][1]
				coords -= np.mean(coords,axis=0)
				tore.mols.append(Mol(atoms,coords))  #[np.argsort(perm)],coords[np.argsort(perm)]))
		tore.WriteXYZ("scans")
		return tore

	def DihedralSamples(self,mol,nrand = 2000):
		tore = MSet("Dihedrals")
		natom = mol.NAtoms()
		for i in range(nrand):
			perm = mol.GreedyDihedOrdering()
			self.read_cartesian(mol.atoms[perm], mol.coords[perm])
			import MolEmb
			#print(MolEmb.Make_DistMat(mol.coords[perm]))
			self.cartesian_to_zmatrix()
			#print(self.zmatrix)
			rnd = np.random.normal(scale=Pi/6.,size=(natom-3))
			for j in range(3,natom):
				self.zmatrix[j][1][2][1] += rnd[j-3]
				# Only tweak a dihedral if there is 1,2,3,4 connectivity.
				if 0:
					atomb = self.zmatrix[j][1][0][0]
					atomc = self.zmatrix[j][1][1][0]
					atomd = self.zmatrix[j][1][2][0]
					print((self.zmatrix[j][1][0][1] < 1.5),np.linalg.norm(self.cartesian[atomb][1]-self.cartesian[atomc][1]) < 1.5)
					connectivity = ((self.zmatrix[j][1][0][1] < 1.5) and np.linalg.norm(self.cartesian[atomb][1]-self.cartesian[atomc][1]) < 1.5)
					if (connectivity):
						self.zmatrix[j][1][2][1] += rnd[j-3]
			self.zmatrix_to_cartesian()
			from ..Containers.Mol import Mol
			atoms = np.array([c[0] for c in self.cartesian],dtype=np.int32)
			coords = np.zeros((natom,3))
			for j in range(natom):
				coords[j] = self.cartesian[j][1]
			coords -= np.mean(coords,axis=0)
			tore.mols.append(Mol(atoms[np.argsort(perm)],coords[np.argsort(perm)]))  #[np.argsort(perm)],coords[np.argsort(perm)]))
		tore.WriteXYZ("InitalConfs")
		return tore

	def read_zmatrix(self, atoms, coords):
		'''
		The z-matrix is a list of triples in the form
		[ name, [[ atom1, distance ], [ atom2, angle ], [ atom3, dihedral ]], mass ]
		The first three atoms have blank lists for the undefined coordinates
		'''
		self.zmatrix = []
		for i,c in enumerate(coords):
			if i == 0:
				self.zmatrix.append([atoms[i],[],[],[]])
			elif i==1:
				self.zmatrix.append([atoms[i],[coords[i][0],coords[i][1]],[],[]])
			elif i==2:
				self.zmatrix.append([atoms[i],[coords[i][0],coords[i][1]],[coords[i][2],coords[i][3]],[]])
			else:
				self.zmatrix.append([atoms[i],[coords[i][0],coords[i][1]],[coords[i][2],coords[i][3]],[coords[i][4],coords[i][5]]])
		return self.zmatrix

	def read_cartesian(self, atoms, coords):
		'''Read the cartesian coordinates file (assumes no errors)'''
		'''The cartesian coordiantes consist of a list of atoms formatted as follows
		[ name, np.array( [ x, y, z ] ), mass ]
		'''
		self.cartesian = []
		for i,c in enumerate(coords):
			self.cartesian.append([atoms[i],coords[i]])
		return self.cartesian

	def rotation_matrix(self, axis, angle):
		'''Euler-Rodrigues formula for rotation matrix'''
		# Normalize the axis
		axis = axis / np.sqrt(np.dot(axis, axis))
		a = np.cos(angle / 2)
		b, c, d = -axis * np.sin(angle / 2)
		return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
			[2 * (b * c + a * d), a * a + c * c -b * b - d * d,  2 * (c * d - a * b)],
			[2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

	def add_first_three_to_cartesian(self):
		'''The first three atoms in the zmatrix need to be treated differently'''
		# First atom
		name, coords = self.zmatrix[0]
		self.cartesian = [[name, np.array([0, 0, 0])]]
		# Second atom
		name, coords = self.zmatrix[1]
		distance = coords[0][1]
		self.cartesian.append(
			[name, np.array([distance, 0, 0])])
		# Third atom
		name, coords = self.zmatrix[2]
		atom1, atom2 = coords[:2]
		atom1, distance = atom1
		atom2, angle = atom2
		q = np.array(self.cartesian[atom1][1])  # position of atom 1
		r = np.array(self.cartesian[atom2][1])  # position of atom 2
		# Vector pointing from q to r
		a = r - q
		# Vector of length distance pointing along the x-axis
		d = distance * a / np.sqrt(np.dot(a, a))
		# Rotate d by the angle around the z-axis
		d = np.dot(self.rotation_matrix([0, 0, 1], angle), d)
		# Add d to the position of q to get the new coordinates of the atom
		p = q + d
		atom = [name, p]
		self.cartesian.append(atom)

	def add_atom_to_cartesian(self, coords):
		'''Find the cartesian coordinates of the atom'''
		name, coords = coords
		atom1, distance = coords[0]
		atom2, angle = coords[1]
		atom3, dihedral = coords[2]
		q = self.cartesian[atom1][1]  # atom 1
		r = self.cartesian[atom2][1]  # atom 2
		s = self.cartesian[atom3][1]  # atom 3
		# Vector pointing from q to r
		a = r - q
		# Vector pointing from s to r
		b = r - s
		# Vector of length distance pointing from q to r
		d = distance * a / np.sqrt(np.dot(a, a))
		# Vector normal to plane defined by q,r,s
		normal = np.cross(a, b)
		# Rotate d by the angle around the normal to the plane defined by q,r,s
		d = np.dot(self.rotation_matrix(normal, angle), d)
		# Rotate d around a by the dihedral
		d = np.dot(self.rotation_matrix(a, dihedral), d)
		# Add d to the position of q to get the new coordinates of the atom
		p = q + d
		atom = [name, p]
		self.cartesian.append(atom)

	def zmatrix_to_cartesian(self):
		'''Convert the zmartix to cartesian coordinates'''
		# Deal with first three line separately
		self.cartesian = []
		self.add_first_three_to_cartesian()
		for atom in self.zmatrix[3:]:
			self.add_atom_to_cartesian(atom)
		self.remove_dummy_atoms()
		return self.cartesian

	def add_first_three_to_zmatrix_old(self):
		'''The first three atoms need to be treated differently'''
		# First atom
		self.zmatrix = []
		name, position = self.cartesian[0]
		self.zmatrix.append([name, [[0,0], [0,0], [0,0]]])

		# Second atom
		name, position = self.cartesian[1]
		atom1 = self.cartesian[0]
		pos1 = atom1[1]
		q = pos1 - position
		distance = m.sqrt(np.dot(q, q))
		self.zmatrix.append([name, [[0, distance], [0,0], [0,0]]])

		# Third atom
		name, position = self.cartesian[2]
		atom1, atom2 = self.cartesian[:2]
		pos1, pos2 = atom1[1], atom2[1]
		q = pos1 - position
		r = pos2 - pos1
		q_u = q / np.sqrt(np.dot(q, q))
		r_u = r / np.sqrt(np.dot(r, r))
		distance = np.sqrt(np.dot(q, q))
		# Angle between a and b = acos( dot product of the unit vectors )
		angle = m.acos(np.dot(-q_u, r_u))
		self.zmatrix.append(
		[name, [[0, distance], [1, angle], [0,0]]])

	def add_first_three_to_zmatrix(self):
		'''The first three atoms need to be treated differently'''
		# First atom
		self.zmatrix = []
		name, position = self.cartesian[0]
		self.zmatrix.append([name, [[0,0], [0,0], [0,0]]])

		# Second atom
		name, position = self.cartesian[1]
		atom1 = self.cartesian[0]
		pos1 = atom1[1]
		q = pos1 - position
		distance = m.sqrt(np.dot(q, q))
		self.zmatrix.append([name, [[0, distance], [0,0], [0,0]]])

		# Third atom
		name, position = self.cartesian[2]
		atom1, atom2 = self.cartesian[:2]
		pos1, pos2 = atom1[1], atom2[1]
		q = pos1 - position
		r = pos2 - pos1
		q_u = q / np.sqrt(np.dot(q, q))
		r_u = r / np.sqrt(np.dot(r, r))
		distance = np.sqrt(np.dot(q, q))
		# Angle between a and b = acos( dot product of the unit vectors )
		angle = m.acos(np.dot(-q_u, r_u))
		self.zmatrix.append(
		[name, [[0, distance], [1, angle], [0,0]]])

	def add_atom_to_zmatrix(self, i, line):
		'''Generates an atom for the zmatrix
		(assumes that three previous atoms have been placed in the cartesian coordiantes)'''
		name, position = line
		atom1, atom2, atom3 = self.cartesian[:3]
		pos1, pos2, pos3 = atom1[1], atom2[1], atom3[1]
		# Create vectors pointing from one atom to the next
		q = pos1 - position
		r = pos2 - pos1
		s = pos3 - pos2
		position_u = position / np.sqrt(np.dot(position, position))
		# Create unit vectors
		q_u = q / np.sqrt(np.dot(q, q))
		r_u = r / np.sqrt(np.dot(r, r))
		s_u = s / np.sqrt(np.dot(s, s))
		distance = np.sqrt(np.dot(q, q))
		# Angle between a and b = acos( dot( a, b ) / ( |a| |b| ) )
		angle = m.acos(np.dot(-q_u, r_u))
		angle_123 = m.acos(np.dot(-r_u, s_u))
		# Dihedral angle = acos( dot( normal_vec1, normal_vec2 ) / (
		# |normal_vec1| |normal_vec2| ) )
		plane1 = np.cross(q, r)
		plane2 = np.cross(r, s)
		dihedral = m.acos(np.dot(
			plane1, plane2) / (np.sqrt(np.dot(plane1, plane1)) * np.sqrt(np.dot(plane2, plane2))))
		# Convert to signed dihedral angle
		if np.dot(np.cross(plane1, plane2), r_u) < 0:
			dihedral = -dihedral
		coords = [[0, distance], [1, angle], [
			2, dihedral]]
		atom = [name, coords]
		self.zmatrix.append(atom)

	def cartesian_to_zmatrix(self):
		'''Convert the cartesian coordinates to a zmatrix'''
		self.zmatrix=[]
		self.add_first_three_to_zmatrix()
		for i, atom in enumerate(self.cartesian[3:], start=3):
			self.add_atom_to_zmatrix(i, atom)
		return self.zmatrix

	def remove_dummy_atoms(self):
		'''Delete any dummy atoms that may have been placed in the calculated cartesian coordinates'''
		new_cartesian = []
		for line in self.cartesian:
			if not line[0] == 'X':
				new_cartesian.append(line)
		self.cartesian = new_cartesian

	def XYZtoZ(self,atoms,coords):
		self.read_cartesian(atoms,coords)
		self.cartesian_to_zmatrix()
		return np.array([c[0] for c in self.zmatrix],dtype=np.int32), np.array([[c[1][0][0],c[1][0][1],c[1][1][0],c[1][1][1],c[1][2][0],c[1][2][1]] for c in self.zmatrix],dtype=np.float64)

	def ZtoXYZ(self,atoms,coords):
		self.read_zmat(atoms,coords)
		self.zmatrix_to_cartesian()
		return np.array([c[0] for c in self.cartesian],dtype=np.int32), np.array([c[1:4] for c in self.cartesian],dtype=np.float64)
