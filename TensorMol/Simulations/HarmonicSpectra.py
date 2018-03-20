from __future__ import absolute_import
from __future__ import print_function
from .SimpleMD import *
from .Opt import *
from ..Math import *

def HarmonicSpectra(f_, x_, at_, grad_=None, eps_ = 0.001, WriteNM_=False, Mu_ = None, Temp_=300.0, Pres_=101325.0, h_=None):
	"""
	Perform a finite difference normal mode analysis
	of a molecule. basically implements http://gaussian.com/vib/
	also does a thermodynamic analysis at 300K.

	Args:
		f_: Energies in Hartree.
		x_: Coordinates (A)
		at_: element type of each atom
		grad_: forces in Hartree/angstrom if available. (unused)
		eps_: finite difference step
		WriteNM_: Whether to write the normal modes to readable files
		Mu_: A dipole field routine for intensities. (atomic units)
		Temp_: Non-standard temperature (Kelvin)
		Pressure_: Non-standard Pressure (Pascals)
		h_: analytical Hessian (if available)

	Returns:
		Frequencies in wavenumbers, Normal modes (cart), and Intensities, and Thermodynamic dictionary.
	"""
	LOGGER.info("Harmonic Analysis")
	n = x_.shape[0]
	n3 = 3*n
	# m_ is a mass vector in atomic units.
	m_ = np.array(list(map(lambda x: ATOMICMASSESAMU[x-1]*ELECTRONPERPROTONMASS, at_.tolist())))
	print ("m_:", m_)
	Crds = InternalCoordinates(x_,m_) #invbasis X cart
	#Crds=np.eye(n3).reshape((n3,n,3))
	#print("En?",f_(x_))
	E0 = f_(x_)
	if 0:
		Hess = DirectedFdiffHessian(f_, x_, Crds.reshape((len(Crds),n,3)))
		print("Hess (Internal):", Hess)
		# Transform the invariant hessian into cartesian coordinates.
		cHess = np.dot(Crds.T,np.dot(Hess,Crds))
	elif (h_ != None):
		print("Calculating Analytical Hessian...")
		cHess = h_(x_).reshape((n3,n3)) # If an analytical hessian is available.
	else:
		cHess = FdiffHessian(f_, x_, eps_).reshape((n3,n3))
	cHess /= (BOHRPERA*BOHRPERA)
	#print("Hess (Cart):", cHess)
	# Mass weight the invariant hessian in cartesian coordinate
	for i,mi in enumerate(m_):
		cHess[i*3:(i+1)*3, i*3:(i+1)*3] /= np.sqrt(mi*mi)
		for j,mj in enumerate(m_):
			if (i != j):
				cHess[i*3:(i+1)*3, j*3:(j+1)*3] /= np.sqrt(mi*mj)
	# Get the vibrational spectrum and normal modes.
	#u,s,v = np.linalg.svd(cHess)
	#for l in s:
	#	print("Central Energy (cm**-1): ", np.sign(l)*np.sqrt(l)*WAVENUMBERPERHARTREE)
	#print("--")

	# Get the actual normal modes, for visualization sake.
	# Toss out the last six modes.
	w,v = np.linalg.eigh(cHess)
	v = v.real
	#print("N3, shape v",n3,v.shape)
	# Perform a thermodynamic analysis.

	# At 300K in Hartrees
	# Everything is done in Atomic Units before conversion.
	KbT = KAYBEEAU*Temp_
	RT = AVOCONST*KbT
	R = RT/Temp_
	Pressure = Pres_/PASCALPERAU
	TotalMass = np.sum(m_)

	# Compute the molecular partition function.
	lambdaDB = np.sqrt(1.0/(2.*Pi*TotalMass*RT))
	qtrans = (RT/Pressure)/(lambdaDB*lambdaDB*lambdaDB) # This is for a mol
	Strans = R*(2.5 + np.log(qtrans))
	Cvtrans = R*1.5
	Cptrans = R*2.5
	Etrans = R*1.5*Temp_
	Htrans = R*2.5*Temp_
	Gtrans = Htrans-Temp_*Strans

	COM0 = CenterOfMass(x_,m_)
	xc_  = x_ - COM0
	xc_ *= BOHRPERA
	I = InertiaTensor(xc_,m_)
	Ic,Raxes = np.linalg.eig(I)
	#print("Rotational Inertia axes: ", Ic/R)
	RCONSTS = 1.0/(8.0*Pi*Pi*Ic)
	qrot = np.sqrt(Pi)*np.sqrt(np.prod(RT/RCONSTS));
	Srot = R*(1.5 + np.log(qrot))
	Cvrot = R*1.5
	Cprot = R*1.5
	Erot = R*1.5*Temp_
	Grot = Erot-Temp_*Srot

	ordering = np.argsort(w)
	VibEs = np.sqrt(np.abs(w[ordering][6:]))
	VibTemps = VibEs/(KAYBEEAU*AVOCONST)+1.0e-36
	ZeroPointEnergy = np.sum(VibEs)/2.0
	rT = VibTemps/Temp_
	Evib = np.sum(VibEs*(0.5+1.0/(np.exp(rT)-1.0)))
	Svib = R*np.sum(rT/(np.exp(rT) - 1.0) - np.log(1.0 - np.exp(-rT)))
	Cvvib = R*np.sum(np.exp(rT)*np.power(rT/(np.exp(rT)-1.0), 2.0))
	Gvib = Evib - Temp_*Svib

	GTotal = E0 + Gvib + Grot + Gtrans

	thermodynamics = {"ElectronicEnergy":E0,"RotEnthalpy":Erot,
						"TransEnthalpy":Htrans,"VibEnthalpy":Evib,
						"Grot":Grot,"Gtrans":Gtrans,
						"Srot":Srot,"Svib":Svib,"Erot":Erot,
						"Gvib":Gvib,"TotalFreeEnergy":GTotal,"Strans":Strans}

	if (WriteNM_):
		intensities = np.zeros(shape=(3*n-6))
		for i in range(6,3*n):
			ip = ordering[i]
			nm = np.zeros(3*n)
			for j,mi in enumerate(m_):
				nm[3*j:3*(j+1)] = v[3*j:3*(j+1),ip]/np.sqrt(mi/ELECTRONPERPROTONMASS)
			#nm /= np.sqrt(np.sum(nm*nm))
			nm = nm.reshape((n,3))
			# Take finite difference derivative of mu(Q) and return the <dmu/dQ, dmu/dQ>
			step = 0.01
			dmudq = (Mu_(x_+step*nm)-Mu_(x_))/step
			print(VibEs[i-6]*WAVENUMBERPERHARTREE,"|f| (UNITS????) ",np.dot(dmudq,dmudq.T))
			intensities[i-6] = np.dot(dmudq,dmudq.T)
			# for alpha in np.append(np.linspace(0.1,-0.1,30),np.linspace(0.1,-0.1,30)):
			# 	mdisp = Mol(at_, x_+alpha*nm)
			# 	#print("Mu",Mu_(x_+alpha*nm))
			# 	mdisp.WriteXYZfile("./results/","NormalMode_"+str(i))
		return VibEs*WAVENUMBERPERHARTREE, v, intensities, thermodynamics
	return VibEs*WAVENUMBERPERHARTREE, v, thermodynamics

def HarmonicSpectraWithProjection(f_, x_, at_, grad_=None, eps_ = 0.001, WriteNM_=False, Mu_ = None):
	"""
	Perform a finite difference normal mode analysis
	of a molecule. basically implements http://gaussian.com/vib/

	Args:
		f_: Energies in Hartree.
		x_: Coordinates (A)
		at_: element type of each atom
		grad_: forces in Hartree/angstrom if available. (unused)
		eps_: finite difference step
		WriteNM_: Whether to write the normal modes to readable files
		Mu_: A dipole field routine for intensities.

	Returns:
		Frequencies in wavenumbers, Normal modes (cart), and Intensities
	"""
	LOGGER.info("Harmonic Analysis")
	n = x_.shape[0]
	n3 = 3*n
	m_ = np.array(map(lambda x: ATOMICMASSESAMU[x-1]*ELECTRONPERPROTONMASS, at_.tolist()))
	Crds = InternalCoordinates(x_,m_) #invbasis X cart flatten.
	cHess = FdiffHessian(f_, x_,0.0005).reshape((n3,n3))
	cHess /= (BOHRPERA*BOHRPERA)
	print("Hess (Cart):", cHess)
	# Mass weight the invariant hessian in cartesian coordinate
	for i,mi in enumerate(m_):
		cHess[i*3:(i+1)*3, i*3:(i+1)*3] /= np.sqrt(mi*mi)
		for j,mj in enumerate(m_):
			if (i != j):
				cHess[i*3:(i+1)*3, j*3:(j+1)*3] /= np.sqrt(mi*mj)
	# Get the vibrational spectrum and normal modes.
	# pHess = np.einsum('ab,cb->ac',np.einsum('ij,jk->ik',Crds,cHess),Crds)
	s,v = np.linalg.eigh(Hess)
	for l in s:
		print("Central Energy (cm**-1): ", np.sign(l)*np.sqrt(l)*WAVENUMBERPERHARTREE)
	print("--")
	# Get the actual normal modes, for visualization sake.
	v = v.real
	wave = np.sign(s)*np.sqrt(abs(s))*WAVENUMBERPERHARTREE
	print("N3, shape v",n3,v.shape)
	if (WriteNM_):
		intensities = np.zeros(shape=(3*n))
		for i in range(3*n):
			nm = np.zeros(3*n)
			for j,mi in enumerate(m_):
				nm[3*j:3*(j+1)] = v[3*j:3*(j+1),i]/np.sqrt(mi/ELECTRONPERPROTONMASS)
			#nm /= np.sqrt(np.sum(nm*nm))
			nm = nm.reshape((n,3))
			# Take finite difference derivative of mu(Q) and return the <dmu/dQ, dmu/dQ>
			step = 0.01
			dmudq = (Mu_(x_+step*nm)-Mu_(x_))/step
			print("|f| (UNITS????) ",np.dot(dmudq,dmudq.T))
			intensities[i] = np.dot(dmudq,dmudq.T)
			# for alpha in np.append(np.linspace(0.1,-0.1,30),np.linspace(0.1,-0.1,30)):
			# 	mdisp = Mol(at_, x_+alpha*nm)
			# 	#print("Mu",Mu_(x_+alpha*nm))
			# 	mdisp.WriteXYZfile("./results/","NormalMode_"+str(i))
		return wave, v, intensities
	return wave, v
