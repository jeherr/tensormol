from __future__ import absolute_import
from __future__ import print_function
from ..Util import *
from ..Math import *
from ..ElementData import *
import MolEmb

class Mol:
	""" Provides a general purpose molecule"""
	def __init__(self, atoms_ =  np.zeros(1,dtype=np.uint8), coords_ = np.zeros(shape=(1,1),dtype=np.float)):
		"""
		Args:
			atoms_: np.array(dtype=uint8) of atomic numbers.
			coords_: np.array(dtype=uint8) of coordinates.
		"""
		self.atoms = atoms_.copy()
		self.coords = coords_.copy()
		self.properties = {}
		self.name=None
		#things below here are sometimes populated if it is useful.
		self.DistMatrix = None # a list of equilbrium distances
		return

	def GenSummary(self):
		"""
		Generates several sorts of properties which
		might be useful in a database. Including hashes
		for connectivity, geometry etc.
		puts these in a summary dictionary which it returns.
		"""
		self.SortAtoms()
		self.BuildDistanceMatrix()
		d,t,q = self.Topology()
		bm = np.zeros((self.NAtoms(),self.NAtoms()))
		for b in d:
			bm[b[0],b[1]] = self.atoms[b[0]]*self.atoms[b[1]]
			bm[b[1],b[0]] = self.atoms[b[0]]*self.atoms[b[1]]
		w,v = np.linalg.eig(bm)
		MW = self.MolecularWeight()
		formula = self.Formula()
		import hashlib
		hasher = hashlib.md5()
		hasher.update(formula.encode('utf-8'))
		hasher.update(str(abs(np.sort(np.round(w,decimals=6)))).encode('utf-8'))
		bondhash = hasher.hexdigest()
		hasher.update(str(np.round(self.DistMatrix,decimals=6)).encode('utf-8'))
		ghash = hasher.hexdigest()
		tore = {"MW":MW, "formula":formula, "bond_hash":bondhash, "geom_hash":ghash}
		for akey in tore.keys():
			self.properties[akey] = tore[akey]
		return tore

	def MolecularWeight(self):
		MW = 0.0
		for atom in self.atoms:
			MW += AtomData[atom][3]
		return MW

	def Formula(self):
		tore = ""
		for element in self.AtomTypes():
			tore = tore + AtomData[element][0]+str(self.NumOfAtomsE(element))
		return tore

	def BuildElectronConfiguration(self,charge_=0,spin_=1):
		"""
		fill up electronic configuration.
		"""
		nelectron = sum(self.atoms) - charge_
		nalpha = (nelectron+spin_)//2
		nbeta = nalpha - self.spin
		basis = []
#		VALENCEBASIS = [[[1,0]],[[1,0]],[[1,0],[2,0]],[[1,0],[2,0]]]
#		for atom in self.atoms:
#			basis.append(VALENCEBASIS[atom])
		self.properties["basis"] = basis
		self.properties["charge"] = charge_
		self.properties["spin"] = spin_
		self.properties["nalpha"] = nalpha
		self.properties["nbeta"] = nbeta
		return

	def AtomTypes(self):
		return np.unique(self.atoms)

	def Num_of_Heavy_Atom(self):
		return len([1 for i in self.atoms if i!=1])

	def NEles(self):
		return len(self.AtomTypes())

	def IsIsomer(self,other):
		return np.array_equals(np.sort(self.atoms),np.sort(other.atoms))

	def NAtoms(self):
		return self.atoms.shape[0]

	def NumOfAtomsE(self, e):
		"""
		Number of atoms of a given Element
		"""
		return sum( [1 if at==e else 0 for at in self.atoms ] )

	def CalculateAtomization(self):
		"""
		This routine needs to be revised and replaced.
		"""
		if ("roomT_H" in self.properties):
			AE = self.properties["roomT_H"]
			for i in range (0, self.atoms.shape[0]):
				if (self.atoms[i] in ELEHEATFORM):
					AE = AE - ELEHEATFORM[self.atoms[i]]
			self.properties["atomization"] = AE
		elif ("energy" in self.properties):
			AE = self.properties["energy"]
			for i in range (0, self.atoms.shape[0]):
				if (self.atoms[i] in ele_avg_wb97xd):
					AE = AE - ele_avg_wb97xd[self.atoms[i]]
			self.properties["atomization"] = AE
		else:
			raise Exception("Missing energy to calculate atomization... ")
		return

	def Rotate(self, axis, ang, origin=np.array([0.0, 0.0, 0.0])):
		"""
		Rotate atomic coordinates and forces if present.

		Args:
			axis: vector for rotation axis
			ang: radians of rotation
			origin: origin of rotation axis.
		"""
		rm = RotationMatrix(axis, ang)
		crds = np.copy(self.coords)
		crds -= origin
		for i in range(len(self.coords)):
			self.coords[i] = np.dot(rm,crds[i])
		if ("forces" in self.properties.keys()):
			# Must also rotate the force vectors
			old_endpoints = crds+self.properties["forces"]
			new_forces = np.zeros(crds.shape)
			for i in range(len(self.coords)):
				new_endpoint = np.dot(rm,old_endpoints[i])
				new_forces[i] = new_endpoint - self.coords[i]
			self.properties["forces"] = new_forces
		self.coords += origin

	def RotateRandomUniform(self, randnums=None, origin=np.array([0.0, 0.0, 0.0])):
		"""
		Rotate atomic coordinates and forces if present.

		Args:
			randnums: theta, phi, and z for rotation, if None then rotation is random
			origin: origin of rotation axis.
		"""
		rm = RotationMatrix_v2(randnums)
		crds = np.copy(self.coords)
		crds -= origin
		self.coords = np.einsum("ij,kj->ki",rm, crds)
		if ("forces" in self.properties.keys()):
			# Must also rotate the force vectors
			old_endpoints = crds+self.properties["forces"]
			new_endpoint = np.einsum("ij,kj->ki",rm, old_endpoints)
			new_forces = new_endpoint - self.coords
			self.properties["forces"] = new_forces
		if ("mmff94forces" in self.properties.keys()):
			# Must also rotate the force vectors
			old_endpoints = crds+self.properties["mmff94forces"]
			new_endpoint = np.einsum("ij,kj->ki",rm, old_endpoints)
			new_forces = new_endpoint - self.coords
			self.properties["mmff94forces"] = new_forces
		self.coords += origin

	def Transform(self,ltransf,center=np.array([0.0,0.0,0.0])):
		crds=np.copy(self.coords)
		for i in range(len(self.coords)):
			self.coords[i] = np.dot(ltransf,crds[i]-center) + center

	def AtomsWithin(self,rad, pt):
		# Returns indices of atoms within radius of point.
		dists = map(lambda x: np.linalg.norm(x-pt),self.coords)
		return [i for i in range(self.NAtoms()) if dists[i]<rad]

	def Distort(self,disp=0.38,movechance=.20):
		''' Randomly distort my coords, but save eq. coords first '''
		self.BuildDistanceMatrix()
		for i in range(0, self.atoms.shape[0]):
			for j in range(0, 3):
				if (random.uniform(0, 1)<movechance):
					#only accept collisionless moves.
					accepted = False
					maxiter = 100
					while (not accepted and maxiter>0):
						tmp = self.coords
						tmp[i,j] += np.random.normal(0.0, disp)
						mindist = np.min([ np.linalg.norm(tmp[i,:]-tmp[k,:]) if i!=k else 1.0 for k in range(self.NAtoms()) ])
						if (mindist>0.35):
							accepted = True
							self.coords = tmp
						maxiter=maxiter-1

	def DistortAN(self,movechance=.15):
		''' Randomly replace atom types. '''
		for i in range(0, self.atoms.shape[0]):
			if (random.uniform(0, 1)<movechance):
				self.atoms[i] = random.random_integers(1,PARAMS["MAX_ATOMIC_NUMBER"])

	def read_xyz_with_properties(self, path, properties, center=True):
		try:
			f=open(path,"r")
			lines=f.readlines()
			natoms=int(lines[0])
			self.atoms.resize((natoms))
			self.coords.resize((natoms,3))
			for i in range(natoms):
				line = lines[i+2].split()
				self.atoms[i]=AtomicNumber(line[0])
				for j in range(3):
					try:
						self.coords[i,j]=float(line[j+1])
					except:
						self.coords[i,j]=scitodeci(line[j+1])
			if center:
				self.coords -= self.Center()
			properties_line = lines[1]
			for i, mol_property in enumerate(properties):
				if mol_property == "name":
					self.properties["name"] = properties_line.split(";")[i]
				if mol_property == "energy":
					self.properties["energy"] = float(properties_line.split(";")[i])
					self.CalculateAtomization()
				if mol_property == "gradients":
					self.properties['gradients'] = np.zeros((natoms, 3))
					read_forces = (properties_line.split(";")[i]).split(",")
					for j in range(natoms):
						for k in range(3):
							self.properties['gradients'][j,k] = float(read_forces[j*3+k])
				if mol_property == "dipole":
					self.properties['dipole'] = np.zeros((3))
					read_dipoles = (properties_line.split(";")[i]).split(",")
					for j in range(3):
						self.properties['dipole'][j] = float(read_dipoles[j])
				if mol_property == "charges":
					self.properties["charges"] = np.zeros((natoms))
					read_charges = (properties_line.split(";")[i]).split(",")
					for j in range(natoms):
						self.properties["charges"][j] = float(read_charges[j])
			f.close()
		except Exception as Ex:
			print("Read Failed.", Ex)
			raise Ex
		return

	def Clean(self):
		self.DistMatrix = None

	def ParseProperties(self,s_):
		"""
		The format of a property string is
		Comment: PropertyName1 Array ;PropertyName2 Array;
		The Property names and contents cannot contain ; :
		"""
		t = s_.split("Comment:")
		t2 = t[1].split(";;;")
		tore = {}
		for prop in t2:
			s = prop.split()
			if (len(s)<1):
				continue
			elif (s[0]=='energy'):
				tore["energy"] = float(s[1])
			elif (s[0]=='Lattice'):
				tore["Lattice"] = np.fromstring(s[1]).reshape((3,3))
		return tore

	def PropertyString(self):
		tore = ""
		for prop in self.properties.keys():
			try:
				if (prop == "energy"):
					tore = tore +";;;"+prop+" "+str(self.properties["energy"])
				elif (prop == "Lattice"):
					tore = tore +";;;"+prop+" "+(self.properties[prop]).tostring()
				else:
					tore = tore +";;;"+prop+" "+str(self.properties[prop])
			except Exception as Ex:
				# print "Problem with energy", string
				pass
		return tore

	def FromXYZString(self,string):
		lines = string.split("\n")
		natoms=int(lines[0])
		if (len(lines[1].split())>1):
			try:
				self.properties = self.ParseProperties(lines[1])
			except Exception as Ex:
				print("Problem with energy", Ex)
				pass
		self.atoms.resize((natoms))
		self.coords.resize((natoms,3))
		for i in range(natoms):
			line = lines[i+2].split()
			if len(line)==0:
				return
			self.atoms[i]=AtomicNumber(line[0])
			try:
				self.coords[i,0]=float(line[1])
			except:
				self.coords[i,0]=scitodeci(line[1])
			try:
				self.coords[i,1]=float(line[2])
			except:
				self.coords[i,1]=scitodeci(line[2])
			try:
				self.coords[i,2]=float(line[3])
			except:
				self.coords[i,2]=scitodeci(line[3])
		if ("energy" in self.properties):
			self.CalculateAtomization()
		return

	def __str__(self,wprop=False):
		lines =""
		natom = self.atoms.shape[0]
		if (wprop):
			lines = lines+(str(natom)+"\nComment: "+self.PropertyString()+"\n")
		else:
			lines = lines+(str(natom)+"\nComment: \n")
		for i in range (natom):
			atom_name =  list(atoi.keys())[list(atoi.values()).index(self.atoms[i])]
			if (i<natom-1):
				lines = lines+(atom_name+"   "+str(self.coords[i][0])+ "  "+str(self.coords[i][1])+ "  "+str(self.coords[i][2])+"\n")
			else:
				lines = lines+(atom_name+"   "+str(self.coords[i][0])+ "  "+str(self.coords[i][1])+ "  "+str(self.coords[i][2]))
		return lines

	def __repr__(self):
		return self.__str__()

	def WriteXYZfile(self, fpath=".", fname="mol", mode="a", wprop = False):
		if not os.path.exists(os.path.dirname(fpath+"/"+fname+".xyz")):
			try:
				os.makedirs(os.path.dirname(fpath+"/"+fname+".xyz"))
			except OSError as exc:
				if exc.errno != errno.EEXIST:
					raise
		with open(fpath+"/"+fname+".xyz", mode) as f:
			for line in self.__str__(wprop).split("\n"):
				f.write(line+"\n")

	def XYZtoGridIndex(self, xyz, ngrids = 250,padding = 2.0):
		Max = (self.coords).max() + padding
		Min = (self.coords).min() - padding
		binsize = (Max-Min)/float(ngrids-1)
		x_index = math.floor((xyz[0]-Min)/binsize)
		y_index = math.floor((xyz[1]-Min)/binsize)
		z_index = math.floor((xyz[2]-Min)/binsize)
		#index=int(x_index+y_index*ngrids+z_index*ngrids*ngrids)
		return x_index, y_index, z_index

	def MolDots(self, ngrids = 250 , padding =2.0, width = 2):
		grids = self.MolGrids()
		for i in range (0, self.atoms.shape[0]):
			x_index, y_index, z_index = self.XYZtoGridIndex(self.coords[i])
			for m in range (-width, width):
				for n in range (-width, width):
					for k in range (-width, width):
						index = (x_index)+m + (y_index+n)*ngrids + (z_index+k)*ngrids*ngrids
						grids[index] = atoc[self.atoms[i]]
		return grids

	def Center(self, CenterOf="Atom", MomentOrder = 1.):
		''' Returns the center of atom or mass

		Args:
			CenterOf: Whether to return center of atom position or mass.
			MomentOrder: Option to do nth order moment.
		Returns:
			Center of Atom, Mass, or a higher-order moment.
		'''
		if (CenterOf == "Mass"):
			m = np.array(map(lambda x: ATOMICMASSES[x-1],self.atoms))
			return np.einsum("ax,a->x",np.power(self.coords,MomentOrder),m)/np.sum(m)
		else:
			return np.average(np.power(self.coords,MomentOrder),axis=0)

	def rms(self, m):
		""" Cartesian coordinate difference. """
		err  = 0.0
		for i in range (0, (self.coords).shape[0]):
			err += np.linalg.norm(m.coords[i] - self.coords[i])
		return err/self.coords.shape[0]

	def rms_inv(self, m):
		""" Invariant coordinate difference. """
		mdm = MolEmb.Make_DistMat(self.coords)
		odm = MolEmb.Make_DistMat(m.coords)
		tmp = (mdm-odm)
		return np.sqrt(np.sum(tmp*tmp)/(mdm.shape[0]*mdm.shape[0]))

	def Topology(self,tol_=1.65,nreal_=-1):
		"""
		Returns:
			Bonds: (NBond X 2) array of bonds (uniquely sorted.)
			Bends: (NBend X 3) array of bends (uniquely sorted.)
			Torsions: (NBend X 4) array of torsions (bends which share bonds) (uniquely sorted.)
		"""
		todo = nreal_
		if (nreal_==-1):
			todo = self.NAtoms()
		tore = np.zeros((self.NAtoms(),3))
		NL = MolEmb.Make_NListNaive(self.coords, tol_, todo, True)
		# Determine bonds
		bonds = []
		for i,js in enumerate(NL):
			if (len(js)>0):
				for j in js:
					if (i<j):
						bonds.append([i,j])
		bends = []
		for i,b1 in enumerate(bonds):
			for b2 in bonds[i+1:]:
				if (b1[0] == b2[0]):
					if (b1[1] < b2[1]):
						bends.append([b1[1],b1[0],b2[1]])
					elif (b1[1] > b2[1]):
						bends.append([b2[1],b1[0],b1[1]])
				elif (b1[0] == b2[1]):
					if (b1[1] < b2[0]):
						bends.append([b1[1],b1[0],b2[0]])
					elif (b1[1] > b2[0]):
						bends.append([b2[0],b1[0],b1[1]])
				elif (b1[1] == b2[0]):
					if (b1[0] < b2[1]):
						bends.append([b1[0],b1[1],b2[1]])
					elif (b1[0] > b2[1]):
						bends.append([b2[1],b1[1],b1[0]])
				elif (b1[1] == b2[1]):
					if (b1[0] < b2[0]):
						bends.append([b1[0],b1[1],b2[0]])
					elif (b1[0] > b2[0]):
						bends.append([b2[0],b1[1],b1[0]])
		torsions = []
		for i,b1 in enumerate(bends):
			for b2 in bends[i+1:]:
				if (b1[1]==b2[0] and b1[2]==b2[1]):
					torsions.append(b1+[b2[2]])
				if (b1[0]==b2[1] and b1[1]==b2[2]):
					torsions.append(b2+[b1[2]])
				if (b1[1]==b2[0] and b1[0]==b2[1]):
					torsions.append(b1[::-1]+[b2[2]])
				if (b1[2]==b2[1] and b1[1]==b2[2]):
					torsions.append(b2+[b1[0]])
				# Note the central bend atom cannot be the center of both in a torsion.
		return np.array(bonds,dtype=np.int32),np.array(bends,dtype=np.int32),np.array(torsions,dtype=np.int32)

	def HybMatrix(self,tol_=2.2,nreal_ = -1):
		"""
		This is a predictor of an atom's hybridization for each atom

		returns 3 numbers for each atom (i) based off neighbors (j)
		\sum_j (1-tanh(pi*(r_ij - 2.5)))/2
		And the mean and std-dev of angles in the triples (radians) weighted by
		the same cut-off factors.
		"""
		todo = nreal_
		if (nreal_==-1):
			todo = self.NAtoms()
		tore = np.zeros((self.NAtoms(),3))
		NL = MolEmb.Make_NListNaive(self.coords, tol_+1.5, todo, True)
		for i,js in enumerate(NL):
			Rijs = np.zeros(len(js))
			for k,j in enumerate(js):
				Rijs[k] = (np.linalg.norm(self.coords[i]-self.coords[j]))
			weights = (1.0-np.tanh((Rijs - tol_)*2.0*3.1415))/2.0
			tore[i,0] = np.sum(weights)
			# Now do weighted angles for all triples.
			angles = []
			aweights = []
			for ji in range(len(js)):
				for ki in range(ji+1,len(js)):
					v1 = self.coords[js[ji]]-self.coords[i]
					v2 = self.coords[js[ki]]-self.coords[i]
					ToACos = (np.sum(v1*v2))/(Rijs[ji]*Rijs[ki]+1e-15)
					if (ToACos > 1.0):
						ToACos = 1.0
					elif (ToACos < -1.0):
						ToACos = -1.0
					angles.append(np.arccos(ToACos))
					aweights.append(weights[ji]*weights[ki])
			angles_a = np.array(angles)
			weights_a = np.array(aweights)
			tore[i,1] = np.sum(angles_a*weights_a)/np.sum(weights_a)
			tmp = (angles_a-tore[i,1])
			tore[i,2] = np.sum(tmp*tmp*weights_a)/np.sum(weights_a)
		print ("tore:", tore)
		return tore

	def BondMatrix(self,tol_ = 1.6):
		"""
		Returns a natom x natom matrix representing the bonds
		of this molecule.
		"""
		mdm = MolEmb.Make_DistMat(self.coords)
		return np.where(mdm < 1.5,np.ones(mdm.shape),np.zeros(mdm.shape))

	def DihedralSamples(self,nsamp=1000):
		"""
		Randomly tweaks dihedrals
		"""
		coords0 = self.coords.copy()
		atoms0 = self.atoms.copy()
		perm = self.GreedyOrdering()
		self.atoms,self.coords = self.atoms[perm],self.coords[perm]
		cm0 = MolEmb.Make_DistMat(self.coords)<1.4
		da,t,q = self.Topology()
		d = da.tolist()
		# Now find the longest chain of bonds recursively.
		d1 = [copy.copy(x[0]) for x in d]
		d2 = [copy.copy(x[1]) for x in d]
		tore = MSet("Dihedrals")
		maxiter = 1000000
		iter = 0
		while (len(tore.mols)<nsamp and iter < maxiter):
			bondi = np.random.randint(0,len(d)-1)
			theta = np.random.uniform(-Pi/2,Pi/2)
			#self.coords = coords0.copy()
			I = d[bondi][0]
			J = d[bondi][1]
			L = self.KevinBacon(d1,d2,I,tips=[],bridge = [J])
			R = self.KevinBacon(d1,d2,J,tips=[],bridge = [I])
			if (len(L)<2 or len(R)<2 or len([k for k in L if k in R])>0):
				continue
			axis = self.coords[I]-self.coords[J]
			center = (self.coords[I]+self.coords[J])/2
			Lrot = RotationMatrix(axis,theta)
			Rrot = RotationMatrix(axis,-theta)
			#print(Lrot,Rrot)
			tmp = self.coords.copy()
			self.coords[L] = np.einsum("ij,kj->ik",self.coords[L]-center,Lrot)
			self.coords[R] = np.einsum("ij,kj->ik",self.coords[R]-center,Rrot)
			self.coords -= np.mean(self.coords,axis=0)
			cmp = MolEmb.Make_DistMat(self.coords)<1.4
			iter = iter + 1
			if (np.any(np.not_equal(cm0,cmp))):
				self.coords = tmp.copy()
				continue
			tore.mols.append(Mol(self.atoms[np.argsort(perm)],self.coords[np.argsort(perm)]))
		#tore.WriteXYZ("InitalConfs")
		self.coords=coords0.copy()
		self.atoms=atoms0.copy()
		return tore

	def KevinBacon(self,d1,d2,i,degree=-1,tips=[],bridge=[]):
		"""
		returns all j connected to i given pairwise connectivity d.
		"""
		oldtips = len(tips)
		if len(tips)==0:
			tips.append(i)
		else:
			for t in tips:
				if t in d1:
					for tp in [d2[k] for k in range(len(d1)) if d1[k]==t]:
						if (not tp in bridge and not tp in tips):
							tips.append(tp)
				if t in d2:
					for tp in [d1[k] for k in range(len(d1)) if d2[k]==t]:
						if (not tp in bridge and not tp in tips):
							tips.append(tp)
		if (oldtips == len(tips) or degree==0):
			return tips
		else:
			return self.KevinBacon(d1,d2,i,degree=degree-1,tips=tips,bridge=bridge)

	def GreedyOrdering(self):
		"""
		find a random ordering which puts bonded atoms near each other
		in the ordering returns the permutation but doesn't apply it.
		"""
		perm = np.random.permutation(self.NAtoms())
		tmpcoords = self.coords[perm]
		dm = MolEmb.Make_DistMat(tmpcoords)
		new = []
		old = list(range(self.NAtoms()))
		new.append(old.pop(0))
		while(len(old)):
			found = False
			dists = dm[new[-1]][old]
			bestdists = np.argsort(dists)
			new.append(old.pop(bestdists[0]))
		# Compose these permutations.
		totalp = perm[new]
		return totalp

	def SpanningGrid(self,num=250,pad=4.,Flatten=True, Cubic = True):
		''' Returns a regular grid the molecule fits into '''
		xmin=np.min(self.coords[:,0])-pad
		xmax=np.max(self.coords[:,0])+pad
		ymin=np.min(self.coords[:,1])-pad
		ymax=np.max(self.coords[:,1])+pad
		zmin=np.min(self.coords[:,2])-pad
		zmax=np.max(self.coords[:,2])+pad
		lx = xmax-xmin
		ly = ymax-ymin
		lz = zmax-zmin
		if (Cubic):
			mlen = np.max([lx,ly,lz])
			xmax = xmin + mlen
			ymax = ymin + mlen
			zmax = zmin + mlen
		grids = np.mgrid[xmin:xmax:num*1j, ymin:ymax:num*1j, zmin:zmax:num*1j]
		grids = grids.transpose((1,2,3,0))
		if (not Flatten):
			return grids.rshape()
		grids = grids.reshape((grids.shape[0]*grids.shape[1]*grids.shape[2], grids.shape[3]))
		return grids, (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

	def AddPointstoMolDots(self, grids, points, value, ngrids =250):  # points: x,y,z,prob    prob is in (0,1)
		points = points.reshape((-1,3))  # flat it
		value = value.reshape(points.shape[0]) # flat it
		value = value/value.max()
		for i in range (0, points.shape[0]):
			x_index, y_index, z_index = self.XYZtoGridIndex(points[i])
			index = x_index + y_index*ngrids + z_index*ngrids*ngrids
			if grids[index] <  int(value[i]*250):
				grids[index] = int(value[i]*250)
		return grids

	def SortAtoms(self):
		""" First sorts by element, then sorts by distance to the center of the molecule
			This improves alignment. """
		order = np.argsort(self.atoms)
		self.atoms = self.atoms[order]
		self.coords = self.coords[order,:]
		self.coords = self.coords - self.Center()
		self.ElementBounds = [[0,0] for i in range(self.NEles())]
		for e, ele in enumerate(self.AtomTypes()):
			inblock=False
			for i in range(0, self.NAtoms()):
				if (not inblock and self.atoms[i]==ele):
					self.ElementBounds[e][0] = i
					inblock=True
				elif (inblock and (self.atoms[i]!=ele or i==self.NAtoms()-1)):
					self.ElementBounds[e][1] = i
					inblock=False
					break
		for e in range(self.NEles()):
			blk = self.coords[self.ElementBounds[e][0]:self.ElementBounds[e][1],:].copy()
			dists = np.sqrt(np.sum(blk*blk,axis=1))
			inds = np.argsort(dists)
			self.coords[self.ElementBounds[e][0]:self.ElementBounds[e][1],:] = blk[inds]
		return

	def Interpolation(self,b,n=10):
		tore = []
		for frac in np.linspace(0.0,1.0,n):
			tore.append(Mol(self.atoms,self.coords*frac+b.coords*(1.0-frac)))
		return tore

	def WriteInterpolation(self,b,n=10):
		for i in range(n): # Check the interpolation.
			m=Mol(self.atoms,self.coords*((9.-i)/9.)+b.coords*((i)/9.))
			m.WriteXYZfile(PARAMS["results_dir"], "Interp"+str(n))

	def AlignAtoms(self, m):
		"""
		Align the geometries and atom order of myself and another molecule.
		This alters both molecules, centering them, and also permutes
		their atoms.

		Args:
			m: A molecule to be aligned with me.
		"""
		assert self.NAtoms() == m.NAtoms(), "Number of atoms do not match"
		self.coords -= self.Center()
		m.coords -= m.Center()
		# try to achieve best rotation alignment between them by aligning the second moments of position
		sdm = MolEmb.Make_DistMat(self.coords)
		d = sdm-MolEmb.Make_DistMat(m.coords)
		BestMomentOverlap = np.sum(d*d)
		BestTriple=[0.,0.,0.]
		for a in np.linspace(-Pi,Pi,20):
			for b in np.linspace(-Pi,Pi,20):
				for c in np.linspace(-Pi,Pi,20):
					tmpm = Mol(m.atoms,m.coords)
					tmpm.Rotate([1.,0.,0.],a)
					tmpm.Rotate([0.,1.,0.],b)
					tmpm.Rotate([0.,0.,1.],c)
					d = sdm-MolEmb.Make_DistMat(tmpm.coords)
					lap = np.sum(d*d)
					if ( lap < BestMomentOverlap ):
						BestTriple = [a,b,c]
						BestMomentOverlap = lap
		m.Rotate([1.,0.,0.],BestTriple[0])
		m.Rotate([0.,1.,0.],BestTriple[1])
		m.Rotate([0.,0.,1.],BestTriple[2])
		#print("After centering and Rotation ---- ")
		#print("Self \n"+self.__str__())
		#print("Other \n"+m.__str__())
		self.SortAtoms()
		m.SortAtoms()
		# Greedy assignment
		for e in range(self.NEles()):
			mones = range(self.ElementBounds[e][0],self.ElementBounds[e][1])
			mtwos = range(self.ElementBounds[e][0],self.ElementBounds[e][1])
			assignedmones=[]
			assignedmtwos=[]
			for b in mtwos:
				acs = self.coords[mones]
				tmp = acs - m.coords[b]
				best = np.argsort(np.sqrt(np.sum(tmp*tmp,axis=1)))[0]
				#print "Matching ", m.coords[b]," to ", self.coords[mones[best]]
				#print "Matching ", b," to ", mones[best]
				assignedmtwos.append(b)
				assignedmones.append(mones[best])
				mones = complement(mones,assignedmones)
			self.coords[mtwos] = self.coords[assignedmones]
			m.coords[mtwos] = m.coords[assignedmtwos]
		self.DistMatrix = MolEmb.Make_DistMat(self.coords)
		m.DistMatrix = MolEmb.Make_DistMat(m.coords)
		diff = np.linalg.norm(self.DistMatrix - m.DistMatrix)
		tmp_coords=m.coords.copy()
		tmp_dm = MolEmb.Make_DistMat(tmp_coords)
		k = 0
		steps = 1
		while (k < 2):
			for i in range(m.NAtoms()):
				for j in range(i+1,m.NAtoms()):
					if m.atoms[i] != m.atoms[j]:
						continue
					ir = tmp_dm[i].copy() - self.DistMatrix[i]
					jr = tmp_dm[j].copy() - self.DistMatrix[j]
					irp = tmp_dm[j].copy()
					irp[i], irp[j] = irp[j], irp[i]
					jrp = tmp_dm[i].copy()
					jrp[i], jrp[j] = jrp[j], jrp[i]
					irp -= self.DistMatrix[i]
					jrp -= self.DistMatrix[j]
					if (np.linalg.norm(irp)+np.linalg.norm(jrp) < np.linalg.norm(ir)+np.linalg.norm(jr)):
						k = 0
						perm=range(m.NAtoms())
						perm[i] = j
						perm[j] = i
						tmp_coords=tmp_coords[perm]
						tmp_dm = MolEmb.Make_DistMat(tmp_coords)
						#print(np.linalg.norm(self.DistMatrix - tmp_dm))
						steps = steps+1
				#print(i)
			k+=1
		m.coords=tmp_coords.copy()
		#print("best",tmp_coords)
		#print("self",self.coords)
		self.WriteInterpolation(Mol(self.atoms,tmp_coords),10)
		return Mol(self.atoms,self.coords), Mol(self.atoms,tmp_coords)

# ---------------------------------------------------------------
#  Functions related to energy models and sampling.
#  all this shit should be moved into a "class Calculator"
# ---------------------------------------------------------------

	def BuildDistanceMatrix(self):
		self.DistMatrix = MolEmb.Make_DistMat(self.coords)

	def MakeBonds(self):
		self.BuildDistanceMatrix()
		maxnb = 0
		bonds = []
		for i in range(self.NAtoms()):
			for j in range(i+1,self.NAtoms()):
				if self.DistMatrix[i,j] < 3.0:
					bonds.append([i,j])
		bonds = np.asarray(bonds, dtype=np.int)
		self.properties["bonds"] = bonds
		self.nbonds = bonds.shape[0]
		f=np.vectorize(lambda x: self.atoms[x])
		self.bondtypes = np.unique(f(bonds), axis=0)
		return self.nbonds

	def BondTypes(self):
		return np.unique(self.bonds[:,0]).astype(int)

	def make_neighbors(self, r_cutoff):
		self.neighbor_list = MolEmb.Make_NListNaive(self.coords, r_cutoff, self.NAtoms(), True)
		self.neighbor_list = [sorted(neighbors) for neighbors in self.neighbor_list]

	def nearest_two_neighbors(self):
		self.BuildDistanceMatrix()
		self.nearest_ns = np.argsort(self.DistMatrix, axis=1)[:,1:3]

	def max_neighbors(self):
		return max([len(atom_neighbors) for atom_neighbors in self.neighbor_list])

	def AtomName(self, i):
		return atoi.keys()[atoi.values().index(self.atoms[i])]

	def AllAtomNames(self):
		names=[]
		for i in range (0, self.atoms.shape[0]):
			names.append(atoi.keys()[atoi.values().index(self.atoms[i])])
		return names

	def MultipoleInputs(self):
		"""
			These are the quantities (in Atomic Units)
			which you multiply the atomic charges by (and sum)
			in order to calculate the multipoles of a molecule
			up to PARAMS["EEOrder"]

			Returns:
				(NAtoms X (monopole, dipole x, ... quad x... etc. ))
		"""
		tore = None
		com = self.Center(OfMass=True)
		if (PARAMS["EEOrder"] == 2):
			tore = np.zeros((self.NAtoms,4))
			for i in range(self.NAtoms()):
				tore[i,0] = 1.0
				tore[i,1:] = self.coords[i]-com
		else:
			raise Exception("Implement... ")
		return tore
