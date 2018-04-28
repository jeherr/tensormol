"""
Raw => various descriptors in Tensorflow code.

The Raw format is a batch of rank three tensors.
mol X MaxNAtom X 4
The final dim is atomic number, x,y,z (Angstrom)

https://www.youtube.com/watch?v=h2zgB93KANE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..ForceModifiers.Neighbors import *
from ..Containers.TensorData import *
from ..Math.TFMath import * # Why is this imported here?
from tensorflow.python.client import timeline
import numpy as np
import time
from tensorflow.python.framework import function
if (HAS_TF):
	import tensorflow as tf

def AllTriples(rng):
	"""Returns all possible triples of an input list.

	Args:
		rng: a 1D integer tensor to be triply outer product'd
	Returns:
		A natom X natom X natom X 3 tensor of all triples of entries from rng.
	"""
	rshp = tf.shape(rng)
	natom = rshp[0]
	v1 = tf.tile(tf.reshape(rng,[natom,1]),[1,natom])
	v2 = tf.tile(tf.reshape(rng,[1,natom]),[natom,1])
	v3 = tf.transpose(tf.stack([v1,v2],0),perm=[1,2,0])
	# V3 is now all pairs (nat x nat x 2). now do the same with another to make nat X 3
	v4 = tf.tile(tf.reshape(v3,[natom,natom,1,2]),[1,1,natom,1])
	v5 = tf.tile(tf.reshape(rng,[1,1,natom,1]),[natom,natom,1,1])
	v6 = tf.concat([v4,v5], axis = 3) # All triples in the range.
	return v6

def AllTriplesSet(rng, prec=tf.int32):
	"""Returns all possible triples of integers between zero and natom.

	Args:
		rng: a 1D integer tensor to be triply outer product'd
	Returns:
		A Nmol X natom X natom X natom X 4 tensor of all triples.
	"""
	natom = tf.shape(rng)[1]
	nmol = tf.shape(rng)[0]
	v1 = tf.tile(tf.reshape(rng,[nmol,natom,1]),[1,1,natom])
	v2 = tf.tile(tf.reshape(rng,[nmol,1,natom]),[1,natom,1])
	v3 = tf.transpose(tf.stack([v1,v2],1),perm=[0,2,3,1])
	# V3 is now all pairs (nat x nat x 2). now do the same with another to make nat X 3
	v4 = tf.tile(tf.reshape(v3,[nmol,natom,natom,1,2]),[1,1,1,natom,1])
	v5 = tf.tile(tf.reshape(rng,[nmol,1,1,natom,1]),[1,natom,natom,1,1])
	v6 = tf.concat([v4,v5], axis = 4) # All triples in the range.
	v7 = tf.cast(tf.tile(tf.reshape(tf.range(nmol),[nmol,1,1,1,1]),[1,natom,natom,natom,1]), dtype=prec)
	v8 = tf.concat([v7,v6], axis = -1)
	return v8

def AllDoublesSet(rng, prec=tf.int32):
	"""Returns all possible doubles of integers between zero and natom.

	Args:
		natom: max integer
	Returns:
		A nmol X natom X natom X 3 tensor of all doubles.
	"""
	natom = tf.shape(rng)[1]
	nmol = tf.shape(rng)[0]
	v1 = tf.tile(tf.reshape(rng,[nmol,natom,1]),[1,1,natom])
	v2 = tf.tile(tf.reshape(rng,[nmol,1,natom]),[1,natom,1])
	v3 = tf.transpose(tf.stack([v1,v2],1),perm=[0,2,3,1])
	v4 = tf.cast(tf.tile(tf.reshape(tf.range(nmol),[nmol,1,1,1]),[1,natom,natom,1]),dtype=prec)
	v5 = tf.concat([v4,v3], axis = -1)
	return v5

def AllSinglesSet(rng, prec=tf.int32):
	"""Returns all possible triples of integers between zero and natom.

	Args:
		natom: max integer
	Returns:
		A nmol X natom X 2 tensor of all doubles.
	"""
	natom = tf.shape(rng)[1]
	nmol = tf.shape(rng)[0]
	v1 = tf.reshape(rng,[nmol,natom,1])
	v2 = tf.cast(tf.tile(tf.reshape(tf.range(nmol),[nmol,1,1]),[1,natom,1]), dtype=prec)
	v3 = tf.concat([v2,v1], axis = -1)
	return v3

def ZouterSet(Z):
	"""
	Returns the outer product of atomic numbers for all molecules.

	Args:
		Z: nMol X MaxNAtom X 1 Z tensor
	Returns
		Z1Z2: nMol X MaxNAtom X MaxNAtom X 2 Z1Z2 tensor.
	"""
	zshp = tf.shape(Z)
	Zs = tf.reshape(Z,[zshp[0],zshp[1],1])
	z1 = tf.tile(Zs, [1,1,zshp[1]])
	z2 = tf.transpose(z1,perm=[0,2,1])
	return tf.transpose(tf.stack([z1,z2],axis=1),perm=[0,2,3,1])

def DifferenceVectorsSet(r_,prec = tf.float64):
	"""
	Given a nmol X maxnatom X 3 tensor of coordinates this
	returns a nmol X maxnatom X maxnatom X 3 tensor of Rij
	"""
	natom = tf.shape(r_)[1]
	nmol = tf.shape(r_)[0]
	#ri = tf.tile(tf.reshape(r_,[nmol,1,natom,3]),[1,natom,1,1])
	ri = tf.tile(tf.reshape(tf.cast(r_,prec),[nmol,1,natom*3]),[1,natom,1])
	ri = tf.reshape(ri, [nmol, natom, natom, 3])
	rj = tf.transpose(ri,perm=(0,2,1,3))
	return (ri-rj)

def DifferenceVectorsLinear(B, NZP):
	"""
	B: Nmol X NmaxNAtom X 3 coordinate tensor
	NZP: a index matrix (nzp X 3)
	"""
	Ii = tf.slice(NZP,[0,0],[-1,2])
	Ij = tf.concat([tf.slice(NZP,[0,0],[-1,1]),tf.slice(NZP,[0,2],[-1,1])],1)
	Ri = tf.gather_nd(B,Ii)
	Rj = tf.gather_nd(B,Ij)
	A = Ri - Rj
	return A

def TFSymASet(R, Zs, eleps_, SFPs_, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	nzeta = pshape[1]
	neta = pshape[2]
	ntheta = pshape[3]
	nr = pshape[4]
	nsym = nzeta*neta*ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001

	# atom triples.
	ats = AllTriplesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0,0],[nmol,natom,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,0,1],[nmol,natom,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,0,2],[nmol,natom,natom,natom,1])
	Rk_inds = tf.slice(ats,[0,0,0,0,3],[nmol,natom,natom,natom,1])
	Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	Z1Z2 = ZouterSet(Zs)
	ZPairs = tf.gather_nd(Z1Z2,Rjk_inds) # should have shape nmol X natom3 X 2
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom3,1,2]),tf.reshape(eleps_,[1,1,nelep,2])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.logical_and(tf.not_equal(Ri_inds,Rj_inds),tf.not_equal(Ri_inds,Rk_inds)),[nmol,natom3,1]),[1,1,nelep])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.range(nelep)
	ats = tf.tile(tf.reshape(ats,[nmol,natom3,1,4]),[1,1,nelep,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nelep,1]),[nmol,natom3,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom3 * nelep X 5 (mol, i,j,k,l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	miks = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,3],[nnz,1])],axis=-1)
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	B = tf.gather_nd(Rij,miks)
	RijRik = tf.reduce_sum(A*B,axis=1)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)
	RikRik = tf.sqrt(tf.reduce_sum(B*B,axis=1)+infinitesimal)
	denom = RijRij*RikRik+infinitesimal
	# Mask any troublesome entries.
	ToACos = RijRik/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar,ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar,ToACos)
	Thetaijk = tf.acos(ToACos)
	zetatmp = tf.cast(tf.reshape(SFPs_[0],[1,nzeta,neta,ntheta,nr]),prec)
	thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[2],[1,nzeta,neta,ntheta,nr]),[nnz,1,1,1,1]),prec)
	# Broadcast the thetas and ToCos together
	tct = tf.tile(tf.reshape(Thetaijk,[nnz,1,1,1,1]),[1,nzeta,neta,ntheta,nr])
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zetatmp)*tf.pow((1.0+Tijk),zetatmp)
	etmp = tf.cast(tf.reshape(SFPs_[1],[1,nzeta,neta,ntheta,nr]),prec) # ijk X zeta X eta ....
	rtmp = tf.cast(tf.reshape(SFPs_[3],[1,nzeta,neta,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij+RikRik)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnz,1,1,1,1]),[1,nzeta,neta,ntheta,nr]) - rtmp
	ToExp2 = etmp*tet*tet
	ToExp3 = tf.where(tf.greater(ToExp2,30),-30.0*tf.ones_like(ToExp2),-1.0*ToExp2)
	fac2 = tf.exp(ToExp3)
	# And finally the last two factors
	fac3 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros_like(RijRij, dtype=prec),0.5*(tf.cos(3.14159265359*RijRij/R_cut)+1.0))
	fac4 = tf.where(tf.greater_equal(RikRik,R_cut),tf.zeros_like(RikRik, dtype=prec),0.5*(tf.cos(3.14159265359*RikRik/R_cut)+1.0))
	# assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnz,1,1,1,1]),[1,nzeta,neta,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnz*nzeta*neta*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	jk2 = tf.add(tf.multiply(tf.slice(GoodInds,[0,2],[nnz,1]), natom), tf.slice(GoodInds,[0,3],[nnz, 1]))
	mil_jk2 = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,4],[nnz,1]),tf.reshape(jk2,[nnz,1])],axis=-1)
	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnz,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.range(nzeta), neta*ntheta*nr),[nzeta,1]),[1,neta])
	p2_2 = tf.tile(tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.multiply(tf.range(neta),ntheta*nr),[1,neta]),[nzeta,1])],axis=-1),[nzeta,neta,1,2]),[1,1,ntheta,1])
	p3_2 = tf.tile(tf.reshape(tf.concat([p2_2,tf.tile(tf.reshape(tf.multiply(tf.range(ntheta),nr),[1,1,ntheta,1]),[nzeta,neta,1,1])],axis=-1),[nzeta,neta,ntheta,1,3]),[1,1,1,nr,1])
	p4_2 = tf.reshape(tf.concat([p3_2,tf.tile(tf.reshape(tf.range(nr),[1,1,1,nr,1]),[nzeta,neta,ntheta,1,1])],axis=-1),[1,nzeta,neta,ntheta,nr,4])
	p5_2 = tf.reshape(tf.reduce_sum(p4_2,axis=-1),[1,nsym,1]) # scatter_nd only supports upto rank 5... so gotta smush this...
	p6_2 = tf.tile(p5_2,[nnz,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,[nmol,natom,nelep,natom2,nsym])
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymASet_Update(R, Zs, eleps_, SFPs_, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	nzeta = pshape[1]
	neta = pshape[2]
	ntheta = pshape[3]
	nr = pshape[4]
	nsym = nzeta*neta*ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001

	# atom triples.
	ats = AllTriplesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0,0],[nmol,natom,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,0,1],[nmol,natom,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,0,2],[nmol,natom,natom,natom,1])
	Rk_inds = tf.slice(ats,[0,0,0,0,3],[nmol,natom,natom,natom,1])
	Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	Z1Z2 = ZouterSet(Zs)
	ZPairs = tf.gather_nd(Z1Z2,Rjk_inds) # should have shape nmol X natom3 X 2
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom3,1,2]),tf.reshape(eleps_,[1,1,nelep,2])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.logical_and(tf.not_equal(Ri_inds,Rj_inds),tf.not_equal(Ri_inds,Rk_inds)),[nmol,natom3,1]),[1,1,nelep])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.range(nelep)
	ats = tf.tile(tf.reshape(ats,[nmol,natom3,1,4]),[1,1,nelep,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nelep,1]),[nmol,natom3,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom3 * nelep X 5 (mol, i,j,k,l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	miks = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,3],[nnz,1])],axis=-1)
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	B = tf.gather_nd(Rij,miks)
	RijRik = tf.reduce_sum(A*B,axis=1)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)
	RikRik = tf.sqrt(tf.reduce_sum(B*B,axis=1)+infinitesimal)

	MaskDist1 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	MaskDist2 = tf.where(tf.greater_equal(RikRik,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	MaskDist12 = tf.logical_and(MaskDist1, MaskDist2) # nmol X natom3 X nelep
	GoodInds2 = tf.boolean_mask(GoodInds, MaskDist12)
	nnz2 = tf.shape(GoodInds2)[0]
	mijs2 = tf.slice(GoodInds2,[0,0],[nnz2,3])
	miks2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,3],[nnz2,1])],axis=-1)
	A2 = tf.gather_nd(Rij,mijs2)
	B2 = tf.gather_nd(Rij,miks2)
	RijRik2 = tf.reduce_sum(A2*B2,axis=1)
	RijRij2 = tf.sqrt(tf.reduce_sum(A2*A2,axis=1)+infinitesimal)
	RikRik2 = tf.sqrt(tf.reduce_sum(B2*B2,axis=1)+infinitesimal)

	denom = RijRij2*RikRik2
	# Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar,ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar,ToACos)
	Thetaijk = tf.acos(ToACos)
	zetatmp = tf.cast(tf.reshape(SFPs_[0],[1,nzeta,neta,ntheta,nr]),prec)
	thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[2],[1,nzeta,neta,ntheta,nr]),[nnz2,1,1,1,1]),prec)
	# Broadcast the thetas and ToCos together
	tct = tf.tile(tf.reshape(Thetaijk,[nnz2,1,1,1,1]),[1,nzeta,neta,ntheta,nr], name="tct")
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zetatmp)*tf.pow((1.0+Tijk),zetatmp)
	etmp = tf.cast(tf.reshape(SFPs_[1],[1,nzeta,neta,ntheta,nr]),prec) # ijk X zeta X eta ....
	rtmp = tf.cast(tf.reshape(SFPs_[3],[1,nzeta,neta,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnz2,1,1,1,1]),[1,nzeta,neta,ntheta,nr], name="tet") - rtmp
	fac2 = tf.exp(-etmp*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	# assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnz2,1,1,1,1]),[1,nzeta,neta,ntheta,nr], name="fac34t")
	Gm = tf.reshape(fac1*fac2*fac34t,[nnz2*nzeta*neta*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	jk2 = tf.add(tf.multiply(tf.slice(GoodInds2,[0,2],[nnz2,1]), natom), tf.slice(GoodInds2,[0,3],[nnz2, 1]))
	mil_jk2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,4],[nnz2,1]),tf.reshape(jk2,[nnz2,1])],axis=-1)
	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnz2,1,4]),[1,nsym,1], name="mil_jk_Outer2")
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.range(nzeta), neta*ntheta*nr),[nzeta,1]),[1,neta])
	p2_2 = tf.tile(tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.multiply(tf.range(neta),ntheta*nr),[1,neta]),[nzeta,1])],axis=-1),[nzeta,neta,1,2]),[1,1,ntheta,1])
	p3_2 = tf.tile(tf.reshape(tf.concat([p2_2,tf.tile(tf.reshape(tf.multiply(tf.range(ntheta),nr),[1,1,ntheta,1]),[nzeta,neta,1,1])],axis=-1),[nzeta,neta,ntheta,1,3]),[1,1,1,nr,1])
	p4_2 = tf.reshape(tf.concat([p3_2,tf.tile(tf.reshape(tf.range(nr),[1,1,1,nr,1]),[nzeta,neta,ntheta,1,1])],axis=-1),[1,nzeta,neta,ntheta,nr,4])
	p5_2 = tf.reshape(tf.reduce_sum(p4_2,axis=-1),[1,nsym,1]) # scatter_nd only supports upto rank 5... so gotta smush this...
	p6_2 = tf.tile(p5_2,[nnz2,1,1], name="p6_tile") # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnz2*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,[nmol,natom,nelep,natom2,nsym])
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymRSet(R, Zs, eles_, SFPs_, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	neta = pshape[1]
	nr = pshape[2]
	nsym = neta*nr
	infinitesimal = 0.000000000000000000000000001

	# atom triples.
	ats = AllDoublesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0],[nmol,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,1],[nmol,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,2],[nmol,natom,natom,1])
	#Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	ZAll = AllDoublesSet(Zs)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1]) # should have shape nmol X natom X natom X 1
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom2,1,1]),tf.reshape(eles_,[1,1,nele,1])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.not_equal(Ri_inds,Rj_inds),[nmol,natom2,1]),[1,1,nele])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.range(nele)
	ats = tf.tile(tf.reshape(ats,[nmol,natom2,1,3]),[1,1,nele,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nele,1]),[nmol,natom2,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom2 * nele X 4 (mol, i, j, l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)
	# Mask any troublesome entries.
	etmp = tf.cast(tf.reshape(SFPs_[0],[1,neta,nr]),prec) # ijk X zeta X eta ....
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,neta,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij,[nnz,1,1]),[1,neta,nr]) - rtmp
	fac1 = tf.exp(-etmp*tet*tet)
	# And finally the last two factors
	fac2 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros_like(RijRij, dtype=prec),0.5*(tf.cos(3.14159265359*RijRij/R_cut)+1.0))
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1,1]),[1,neta,nr])
	# assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*neta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,3],[nnz,1]),tf.slice(GoodInds,[0,2],[nnz,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.range(neta), nr),[neta,1,1]),[1,nr,1])
	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.range(nr),[1,nr,1]),[neta,1,1])],axis=-1),[1,neta,nr,2])
	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
	p4_2 = tf.tile(p3_2,[nnz,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,[nmol,natom,nele,natom,nsym])
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymRSet_Update(R, Zs, eles_, SFPs_, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	neta = pshape[1]
	nr = pshape[2]
	nsym = neta*nr
	infinitesimal = 0.000000000000000000000000001

	# atom triples.
	ats = AllDoublesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0],[nmol,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,1],[nmol,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,2],[nmol,natom,natom,1])
	#Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	ZAll = AllDoublesSet(Zs)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1]) # should have shape nmol X natom X natom X 1
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom2,1,1]),tf.reshape(eles_,[1,1,nele,1])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.not_equal(Ri_inds,Rj_inds),[nmol,natom2,1]),[1,1,nele])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.range(nele)
	ats = tf.tile(tf.reshape(ats,[nmol,natom2,1,3]),[1,1,nele,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nele,1]),[nmol,natom2,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom2 * nele X 4 (mol, i, j, l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)

	MaskDist = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	GoodInds2 = tf.boolean_mask(GoodInds, MaskDist)
	nnz2 = tf.shape(GoodInds2)[0]
	mijs2 = tf.slice(GoodInds2,[0,0],[nnz2,3])
	A2 = tf.gather_nd(Rij,mijs2)
	RijRij2 = tf.sqrt(tf.reduce_sum(A2*A2,axis=1)+infinitesimal)

	# Mask any troublesome entries.
	etmp = tf.cast(tf.reshape(SFPs_[0],[1,neta,nr]),prec) # ijk X zeta X eta ....
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,neta,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz2,1,1]),[1,neta,nr]) - rtmp
	fac1 = tf.exp(-etmp*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz2,1,1]),[1,neta,nr])
	# assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz2*neta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,3],[nnz2,1]),tf.slice(GoodInds2,[0,2],[nnz2,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz2,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.range(neta), nr),[neta,1,1]),[1,nr,1])
	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.range(nr),[1,nr,1]),[neta,1,1])],axis=-1),[1,neta,nr,2])
	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
	p4_2 = tf.tile(p3_2,[nnz2,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz2*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,[nmol,natom,nele,natom,nsym])
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymASet_Update2(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001

	# atom triples.
	ats = AllTriplesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]), dtype=tf.int64), prec=tf.int64)
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0,0],[nmol,natom,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,0,1],[nmol,natom,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,0,2],[nmol,natom,natom,natom,1])
	Rk_inds = tf.slice(ats,[0,0,0,0,3],[nmol,natom,natom,natom,1])
	Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	Z1Z2 = ZouterSet(Zs)
	ZPairs = tf.gather_nd(Z1Z2,Rjk_inds) # should have shape nmol X natom3 X 2
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom3,1,2]),tf.reshape(eleps_,[1,1,nelep,2])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.logical_and(tf.not_equal(Ri_inds,Rj_inds),tf.not_equal(Ri_inds,Rk_inds)),[nmol,natom3,1]),[1,1,nelep])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.cast(tf.range(nelep),dtype=tf.int64)
	ats = tf.tile(tf.reshape(ats,[nmol,natom3,1,4]),[1,1,nelep,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nelep,1]),[nmol,natom3,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom3 * nelep X 5 (mol, i,j,k,l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	miks = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,3],[nnz,1])],axis=-1)
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	B = tf.gather_nd(Rij,miks)
	RijRik = tf.reduce_sum(A*B,axis=1)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)
	RikRik = tf.sqrt(tf.reduce_sum(B*B,axis=1)+infinitesimal)

	MaskDist1 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	MaskDist2 = tf.where(tf.greater_equal(RikRik,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	MaskDist12 = tf.logical_and(MaskDist1, MaskDist2) # nmol X natom3 X nelep
	GoodInds2 = tf.boolean_mask(GoodInds, MaskDist12)
	nnz2 = tf.shape(GoodInds2)[0]
	mijs2 = tf.slice(GoodInds2,[0,0],[nnz2,3])
	miks2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,3],[nnz2,1])],axis=-1)
	A2 = tf.gather_nd(Rij,mijs2)
	B2 = tf.gather_nd(Rij,miks2)
	RijRik2 = tf.reduce_sum(A2*B2,axis=1)
	RijRij2 = tf.sqrt(tf.reduce_sum(A2*A2,axis=1)+infinitesimal)
	RikRik2 = tf.sqrt(tf.reduce_sum(B2*B2,axis=1)+infinitesimal)

	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnz2,1,1]),prec)
	# Broadcast the thetas and ToCos together
	tct = tf.tile(tf.reshape(Thetaijk,[nnz2,1,1]),[1,ntheta,nr])
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnz2,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	# assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnz2,1,1]),[1,ntheta,nr])
	#Gm = tf.reshape(fac2*fac34t,[nnz2*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm = tf.reshape(fac1*fac2*fac34t,[nnz2*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	jk2 = tf.add(tf.multiply(tf.slice(GoodInds2,[0,2],[nnz2,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(GoodInds2,[0,3],[nnz2, 1]))
	mil_jk2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,4],[nnz2,1]),tf.reshape(jk2,[nnz2,1])],axis=-1)
	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnz2,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.

	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.cast(tf.range(ntheta), dtype=tf.int64), tf.cast(nr, dtype=tf.int64)),[ntheta,1,1]),[1,nr,1])
	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[1,nr,1]),[ntheta,1,1])],axis=-1),[1,ntheta,nr,2])
	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
	p6_2 = tf.tile(p3_2,[nnz2,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnz2*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nelep,natom2,nsym], dtype=tf.int64))
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)



def TFSymRSet_Update2(R, Zs, eles_, SFPs_, eta, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001

	# atom triples.
	ats = AllDoublesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]), dtype=tf.int64), prec=tf.int64)
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0],[nmol,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,1],[nmol,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,2],[nmol,natom,natom,1])
	#Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	ZAll = AllDoublesSet(Zs, prec=tf.int64)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1]) # should have shape nmol X natom X natom X 1
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom2,1,1]),tf.reshape(eles_,[1,1,nele,1])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.not_equal(Ri_inds,Rj_inds),[nmol,natom2,1]),[1,1,nele])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.cast(tf.range(nele), dtype=tf.int64)
	ats = tf.tile(tf.reshape(ats,[nmol,natom2,1,3]),[1,1,nele,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nele,1]),[nmol,natom2,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom2 * nele X 4 (mol, i, j, l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)

	MaskDist = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	GoodInds2 = tf.boolean_mask(GoodInds, MaskDist)
	nnz2 = tf.shape(GoodInds2)[0]
	mijs2 = tf.slice(GoodInds2,[0,0],[nnz2,3])
	A2 = tf.gather_nd(Rij,mijs2)
	RijRij2 = tf.sqrt(tf.reduce_sum(A2*A2,axis=1)+infinitesimal)

	# Mask any troublesome entries.
	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz2,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz2,1]),[1,nr])
	# assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz2*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,3],[nnz2,1]),tf.slice(GoodInds2,[0,2],[nnz2,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz2,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p2_2 = tf.reshape(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[nr,1]),[1,nr,1])
	p4_2 = tf.tile(p2_2,[nnz2,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz2*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymASet_Linear(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, Angtri, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		Angtri: angular triples within the cutoff.
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(Angtri)[0]

	Z1Z2 = ZouterSet(Zs)

	Rij_inds = tf.slice(Angtri,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(Angtri,[0,0],[nnzt,2]), tf.slice(Angtri,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(Angtri,[0,0],[nnzt,1]), tf.slice(Angtri,[0,2],[nnzt,2])],axis=-1)
	ZPairs = tf.gather_nd(Z1Z2, Rjk_inds)
	EleIndex = tf.slice(tf.where(tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nnzt,1,2]), tf.reshape(eleps_,[1, nelep, 2])),axis=-1)),[0,1],[nnzt,1])
	GoodInds2 = tf.concat([Angtri,EleIndex],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.reshape(ToExp,[nnzt,1,1]) - rtmp
	#tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	jk2 = tf.add(tf.multiply(tf.slice(GoodInds2,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(GoodInds2,[0,3],[nnzt, 1]))
	mil_jk2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnzt,2]),tf.slice(GoodInds2,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnzt,1,4]),[1,nsym,1])
	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.cast(tf.range(ntheta), dtype=tf.int64), tf.cast(nr, dtype=tf.int64)),[ntheta,1,1]),[1,nr,1])
	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[1,nr,1]),[ntheta,1,1])],axis=-1),[1,ntheta,nr,2])
	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
	p6_2 = tf.tile(p3_2,[nnzt,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnzt*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	#to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nelep,natom2,nsym], dtype=tf.int64))  # scatter_nd way to do it
	to_reduce2 = tf.SparseTensor(ind2, Gm, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(natom, tf.int64), tf.cast(nelep, tf.int64), tf.cast(natom2, tf.int64), tf.cast(nsym, tf.int64)])
	#to_reduce2_reorder = tf.sparse_reorder(to_reduce2)
	reduced2 = tf.sparse_reduce_sum_sparse(to_reduce2, axis=3)
	#to_reduce2_dense = tf.sparse_tensor_to_dense(to_reduce2, validate_indices=False)
	#return tf.sparse_reduce_sum(to_reduce2, axis=3)
	#return tf.reduce_sum(to_reduce2_dense, axis=3)
	return tf.sparse_tensor_to_dense(reduced2)

def TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	#thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	#tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
	tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	#jk2 = tf.add(tf.multiply(tf.slice(AngtriEle,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(AngtriEle,[0,3],[nnzt, 1]))
	#mil_jk2 = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]),tf.slice(AngtriEle,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk2,[0,3], [nnzt, 1])) + 1

	Gm2= tf.reshape(Gm, [nnzt, nsym])
	to_reduce2 = tf.scatter_nd(mil_jk2, Gm2, tf.cast([nmol,natom, nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
#	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnzt,1,4]),[1,nsym,1])
#	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
#	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.cast(tf.range(ntheta), dtype=tf.int64), tf.cast(nr, dtype=tf.int64)),[ntheta,1,1]),[1,nr,1])
#	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[1,nr,1]),[ntheta,1,1])],axis=-1),[1,ntheta,nr,2])
#	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
#	p6_2 = tf.tile(p3_2,[nnzt,1,1]) # should be nnz X nsym
#	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnzt*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
#	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom, nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))  # scatter_nd way to do it
#	#to_reduce2 = tf.SparseTensor(ind2, Gm, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(natom, tf.int64), tf.cast(nelep, tf.int64), tf.cast(jk_max, tf.int64), tf.cast(nsym, tf.int64)])
#	#to_reduce2_reorder = tf.sparse_reorder(to_reduce2)
#	#reduced2 = tf.sparse_reduce_sum_sparse(to_reduce2, axis=3)
#	#to_reduce2_dense = tf.sparse_tensor_to_dense(to_reduce2, validate_indices=True)
#	#to_reduce2_dense = tf.sparse_to_dense(ind2, [tf.cast(nmol, tf.int64), tf.cast(natom, tf.int64), tf.cast(nelep, tf.int64), tf.cast(jk_max, tf.int64), tf.cast(nsym, tf.int64)], Gm)
#	#to_reduce2_dense = tf.sparse_to_dense(ind2, [tf.cast(nmol, tf.int64), tf.cast(natom, tf.int64), tf.cast(nelep, tf.int64), tf.cast(natom2, tf.int64), tf.cast(nsym, tf.int64)], Gm, validate_indices=True)
#	#return tf.sparse_reduce_sum(to_reduce2, axis=3)
	return tf.reduce_sum(to_reduce2, axis=3)
	#return tf.sparse_tensor_to_dense(reduced2), ind2


def TFSymASet_Linear_WithEle_Channel(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, channel_eleps, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	#thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	#tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
	tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	#jk2 = tf.add(tf.multiply(tf.slice(AngtriEle,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(AngtriEle,[0,3],[nnzt, 1]))
	#mil_jk2 = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]),tf.slice(AngtriEle,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk2,[0,3], [nnzt, 1])) + 1

	Gm2= tf.reshape(Gm, [nnzt, nsym])

	tomult = tf.cast(tf.reshape(channel_eleps,[1,1,nelep,1]), dtype=tf.float64)

	to_reduce2 = tf.scatter_nd(mil_jk2, Gm2, tf.cast([nmol,natom, nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
	to_reduce3 = tf.reduce_sum(to_reduce2, axis=3)

	to_reduce4 = tomult*to_reduce3

	return tf.reduce_sum(to_reduce4, axis=2)

def TFSymASet_Linear_WithEle_Channel2(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, channel_eleps, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	num_elep, num_dim = eleps_.get_shape().as_list()
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	#thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	#tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
	tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	#jk2 = tf.add(tf.multiply(tf.slice(AngtriEle,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(AngtriEle,[0,3],[nnzt, 1]))
	#mil_jk2 = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]),tf.slice(AngtriEle,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk2,[0,3], [nnzt, 1])) + 1
	Gm2= tf.reshape(Gm, [nnzt, nsym])

	#tomult = tf.cast(tf.reshape(channel_eleps,[1,1,nelep,1]), dtype=tf.float64)
	#to_reduce2 = tf.scatter_nd(mil_jk2, Gm2, tf.cast([nmol,natom, nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
	#to_reduce3 = tf.reduce_sum(to_reduce2, axis=3)
	#to_reduce4 = tomult*to_reduce3
	#return tf.reduce_sum(to_reduce4, axis=2)

	tomult = tf.reshape(tf.gather_nd(channel_eleps, tf.reshape(mil_jk2[:,2],[-1,1])),[-1,1])
	Gm2_mult = tf.cast(tomult, dtype=tf.float64)*Gm2
	to_reduce2  = tf.zeros([nmol, tf.cast(natom, tf.int32), tf.cast(jk_max, tf.int32), nsym], dtype=tf.float64)
	for e in range(num_elep):
		mask_tensor = tf.equal(mil_jk2[:,2],e)
		mil_jk_mask = tf.boolean_mask(mil_jk2, mask_tensor)
		mi_jk_mask =  tf.concat([mil_jk_mask[:,:2],tf.reshape(mil_jk_mask[:,3],[-1,1])], axis=-1)
		Gm2_mult_mask = tf.boolean_mask(Gm2_mult, mask_tensor)
		tmp = tf.scatter_nd(mi_jk_mask, Gm2_mult_mask, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
		to_reduce2 = to_reduce2 + tmp
	return tf.reduce_sum(to_reduce2, axis=2)

def TFSymASet_Linear_WithEle_Channel3(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, channel_eleps, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	num_elep, num_dim = eleps_.get_shape().as_list()
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	#thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	#tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
	tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	#jk2 = tf.add(tf.multiply(tf.slice(AngtriEle,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(AngtriEle,[0,3],[nnzt, 1]))
	#mil_jk2 = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]),tf.slice(AngtriEle,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk2,[0,3], [nnzt, 1])) + 1
	Gm2= tf.reshape(Gm, [nnzt, nsym])

	#tomult = tf.cast(tf.reshape(channel_eleps,[1,1,nelep,1]), dtype=tf.float64)
	#to_reduce2 = tf.scatter_nd(mil_jk2, Gm2, tf.cast([nmol,natom, nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
	#to_reduce3 = tf.reduce_sum(to_reduce2, axis=3)
	#to_reduce4 = tomult*to_reduce3
	#return tf.reduce_sum(to_reduce4, axis=2)

	tomult = tf.reshape(tf.gather_nd(channel_eleps, tf.reshape(mil_jk2[:,2],[-1,1])),[nnzt,-1])

	#hyb_channel_clean = tf.where(tf.is_nan(hyb_channel), tf.zeros_like(hyb_channel, dtype=prec), hyb_channel)

	Gm2_mult = tf.reshape(tf.expand_dims(tomult, 1)*tf.expand_dims(Gm2, 2), [nnzt, -1])

	mi_jk2 =  tf.concat([mil_jk2[:,:2],tf.reshape(mil_jk2[:,3],[-1,1])], axis=-1)
	to_reduce2 = tf.scatter_nd(mi_jk2, Gm2_mult, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(jk_max, tf.int32), nsym*(tf.shape(channel_eleps)[1])], dtype=tf.int64))
	return tf.reduce_sum(to_reduce2, axis=2)

def TFSymASet_Linear_WithEle_ChannelHyb(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, channel_eleps, Hyb, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	num_elep, num_dim = eleps_.get_shape().as_list()
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	#thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	#tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
	tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	#jk2 = tf.add(tf.multiply(tf.slice(AngtriEle,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(AngtriEle,[0,3],[nnzt, 1]))
	#mil_jk2 = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]),tf.slice(AngtriEle,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk2,[0,3], [nnzt, 1])) + 1
	Gm2= tf.reshape(Gm, [nnzt, nsym])

	#tomult = tf.cast(tf.reshape(channel_eleps,[1,1,nelep,1]), dtype=tf.float64)
	#to_reduce2 = tf.scatter_nd(mil_jk2, Gm2, tf.cast([nmol,natom, nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
	#to_reduce3 = tf.reduce_sum(to_reduce2, axis=3)
	#to_reduce4 = tomult*to_reduce3
	#return tf.reduce_sum(to_reduce4, axis=2)

	tomult = tf.reshape(tf.gather_nd(channel_eleps, tf.reshape(mil_jk2[:,2],[-1,1])),[nnzt,-1])

	mj_inds = tf.concat([AngtriEle[:,0], AngtriEle[:,2]],axis=-1)
	mk_inds = tf.concat([AngtriEle[:,0], AngtriEle[:,3]],axis=-1)
	hyb_channel_j  = tf.reshape(tf.gather_nd(Hyb, tf.reshape(mj_inds,[nnzt,2])),[nnzt,-1])+infinitesimal
	hyb_channel_k  = tf.reshape(tf.gather_nd(Hyb, tf.reshape(mk_inds,[nnzt,2])),[nnzt,-1])+infinitesimal
	hyb_channel  = hyb_channel_j*hyb_channel_k/(hyb_channel_j+hyb_channel_k)
	#hyb_channel_clean = tf.where(tf.is_nan(hyb_channel), tf.zeros_like(hyb_channel, dtype=prec), hyb_channel)

	tomul_with_hyb = tf.concat([tomult, hyb_channel],axis=-1)
	Gm2_mult = tf.reshape(tf.expand_dims(tomul_with_hyb, 1)*tf.expand_dims(Gm2, 2), [nnzt, -1])

	mi_jk2 =  tf.concat([mil_jk2[:,:2],tf.reshape(mil_jk2[:,3],[-1,1])], axis=-1)
	to_reduce2 = tf.scatter_nd(mi_jk2, Gm2_mult, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(jk_max, tf.int32), nsym*(tf.shape(channel_eleps)[1]+tf.shape(Hyb)[2])], dtype=tf.int64))
	return tf.reduce_sum(to_reduce2, axis=2)
def TFSymASet_Linear_WithEle_UsingList(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]

	num_elep, num_dim = eleps_.get_shape().as_list()
	elep_range = tf.cast(tf.range(nelep),dtype=tf.int64)

	Asym_ByElep = []
	for e in range(num_elep):
		tomask = tf.equal(AngtriEle[:,4], tf.reshape(elep_range[e], [1,1]))
		AngtriEle_sub = tf.reshape(tf.boolean_mask(AngtriEle, tf.tile(tf.reshape(tomask,[-1,1]),[1,5])),[-1,5])

		tomask1 = tf.equal(mil_jk2[:,2], tf.reshape(elep_range[e], [1,1]))
		mil_jk2_sub = tf.reshape(tf.boolean_mask(mil_jk2, tf.tile(tf.reshape(tomask1,[-1,1]),[1,4])),[-1,4])
		mi_jk2_sub = tf.concat([mil_jk2_sub[:,0:2],  mil_jk2_sub[:,3:]], axis=-1)

		nnzt_sub = tf.shape(AngtriEle_sub)[0]
		Rij_inds = tf.slice(AngtriEle_sub,[0,0],[nnzt_sub,3])
		Rik_inds = tf.concat([tf.slice(AngtriEle_sub,[0,0],[nnzt_sub,2]), tf.slice(AngtriEle_sub,[0,3],[nnzt_sub,1])],axis=-1)
		Rjk_inds = tf.concat([tf.slice(AngtriEle_sub,[0,0],[nnzt_sub,1]), tf.slice(AngtriEle_sub,[0,2],[nnzt_sub,2])],axis=-1)

		Rij = DifferenceVectorsLinear(R, Rij_inds)
		RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
		Rik = DifferenceVectorsLinear(R, Rik_inds)
		RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
		RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
		denom = RijRij2*RikRik2
		#Mask any troublesome entries.
		ToACos = RijRik2/denom
		ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
		ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
		Thetaijk = tf.acos(ToACos)
		thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
		tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
		ToCos = tct-thetatmp
		Tijk = tf.cos(ToCos) # shape: natom3 X ...
		# complete factor 1
		fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
		rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
		ToExp = ((RijRij2+RikRik2)/2.0)
		tet = tf.tile(tf.reshape(ToExp,[nnzt_sub,1,1]),[1,ntheta,nr]) - rtmp
		fac2 = tf.exp(-eta*tet*tet)
		# And finally the last two factors
		fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
		fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
		## assemble the full symmetry function for all triples.
		fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt_sub,1,1]),[1,ntheta,nr])
		Gm = tf.reshape(fac1*fac2*fac34t,[nnzt_sub*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
		jk_max = tf.reduce_max(tf.slice(mil_jk2_sub,[0,3], [nnzt_sub, 1])) + 1
		Gm2= tf.reshape(Gm, [nnzt_sub, nsym])
		to_reduce2 = tf.scatter_nd(mi_jk2_sub, Gm2, tf.cast([nmol,natom, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
		Asym_ByElep.append(tf.reduce_sum(to_reduce2, axis=2))
	return tf.stack(Asym_ByElep, axis=2)


def TFSymASet_Linear_WithElePeriodic(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, nreal, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	#thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	#tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
	tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.reshape(ToExp,[nnzt,1,1]) - rtmp
	#tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t = tf.reshape(fac3*fac4,[nnzt,1,1])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt, nsym]) # nnz X nzeta X neta X ntheta X nr
	#fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	#Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	#jk2 = tf.add(tf.multiply(tf.slice(AngtriEle,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(AngtriEle,[0,3],[nnzt, 1]))
	#mil_jk2 = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]),tf.slice(AngtriEle,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk2,[0,3], [nnzt, 1])) + 1

	Gm2= tf.reshape(Gm, [nnzt, nsym])
	to_reduce2 = tf.scatter_nd(mil_jk2, Gm2, tf.cast([nmol,tf.cast(nreal, tf.int32), nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
	return tf.reduce_sum(to_reduce2, axis=3)

def TFCoulomb(R, Qs, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of sparse-coulomb
	Madelung energy build.

	Args:
	    R: a nmol X maxnatom X 3 tensor of coordinates.
	    Qs : nmol X maxnatom X 1 tensor of atomic charges.
	    R_cut: Radial Cutoff
	    Radpair: None zero pairs X 3 tensor (mol, i, j)
	    prec: a precision.
	Returns:
	    Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=-1)+infinitesimal)
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee


def TFCoulombCosLR(R, Qs, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of long range cutoff sparse-coulomb
	Madelung energy build.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	# Generate LR cutoff Matrix
	Cut = (1.0-0.5*(tf.cos(RijRij2*Pi/R_cut)+1.0))
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2*Cut
	# Scatter Back
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee


def TFCoulombPolyLR(R, Qs, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of short range cutoff sparse-coulomb
	Madelung energy build. Using switch function 1+x^2(2x-3) in http://pubs.acs.org/doi/ipdf/10.1021/ct501131j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	R_width = PARAMS["Poly_Width"]*BOHRPERA
	R_begin = R_cut
	R_end =  R_cut+R_width
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	t = (RijRij2 - R_begin)/R_width
	Cut_step1  = tf.where(tf.greater(t, 0.0), -t*t*(2.0*t-3.0), tf.zeros_like(t))
	Cut = tf.where(tf.greater(t, 1.0), tf.ones_like(t), Cut_step1)
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2*Cut
	# Scatter Back
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee


def TFCoulombPolyLRSR(R, Qs, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of short range and long range cutoff sparse-coulomb
	Madelung energy build. Using switch function 1+x^2(2x-3) in http://pubs.acs.org/doi/ipdf/10.1021/ct501131j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	R_width = PARAMS["Poly_Width"]*BOHRPERA
	R_begin = R_cut
	R_end =  R_cut+R_width
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	t = (RijRij2 - R_begin)/R_width
	Cut_step1  = tf.where(tf.greater(t, 0.0), -t*t*(2.0*t-3.0), tf.zeros_like(t))
	Cut = tf.where(tf.greater(t, 1.0), tf.ones_like(t), Cut_step1)

	R_off = PARAMS["EECutoffOff"]*BOHRPERA
	t_off = (RijRij2 - (R_off-R_width))/R_width
	Cut_off_step1  = tf.where(tf.greater(t_off, 0.0), 1+t_off*t_off*(2.0*t_off-3.0), tf.ones_like(t_off))
	Cut_off  = tf.where(tf.greater(t_off, 1.0), tf.zeros_like(t), Cut_off_step1)
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2*Cut*Cut_off
	# Scatter Back
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee

def TFCoulombEluSRDSFLR(R, Qs, R_cut, Radpair, alpha, elu_a, elu_shift, prec=tf.float64):
	"""
	A tensorflow linear scaling implementation of the Damped Shifted Electrostatic Force with short range cutoff with elu function (const at short range).
	http://aip.scitation.org.proxy.library.nd.edu/doi/pdf/10.1063/1.2206581
	Batched over molecules.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_srcut: Short Range Erf Cutoff
		R_lrcut: Long Range DSF Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		alpha: DSF alpha parameter (~0.2)
	Returns
		Energy of  Mols
	"""
	alpha = alpha/BOHRPERA
	R_lrcut = PARAMS["EECutoffOff"]*BOHRPERA
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	SR_sub = tf.where(tf.greater(RijRij2, R_cut), elu_a*(RijRij2-R_cut)+elu_shift, elu_a*(tf.exp(RijRij2-R_cut)-1.0)+elu_shift)

	twooversqrtpi = tf.constant(1.1283791671,dtype=tf.float64)
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Gather desired LJ parameters.
	Qij = Qi*Qj
	# This is Dan's Equation (18)
	XX = alpha*R_lrcut
	ZZ = tf.erfc(XX)/R_lrcut
	YY = twooversqrtpi*alpha*tf.exp(-XX*XX)/R_lrcut
	LR = Qij*(tf.erfc(alpha*RijRij2)/RijRij2 - ZZ + (RijRij2-R_lrcut)*(ZZ/R_lrcut+YY))
	LR= tf.where(tf.is_nan(LR), tf.zeros_like(LR), LR)
	LR = tf.where(tf.greater(RijRij2,R_lrcut), tf.zeros_like(LR), LR)

	SR = Qij*SR_sub

	K = tf.where(tf.greater(RijRij2, R_cut), LR, SR)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	sparse_index = tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, K, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	# Now use the sparse reduce sum trick to scatter this into mols.
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def TFVdwPolyLR(R, Zs, eles, c6, R_vdw, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of short range cutoff sparse-coulomb
	Madelung energy build. Using switch function 1+x^2(2x-3) in http://pubs.acs.org/doi/ipdf/10.1021/ct501131j
	damping function in http://pubs.rsc.org/en/content/articlepdf/2008/cp/b810189b is used.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		c6 : nele. Grimmer C6 coff in a.u.
		R_vdw: nele. Grimmer vdw radius in a.u.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	R = tf.multiply(R, BOHRPERA)
	R_width = PARAMS["Poly_Width"]*BOHRPERA
	R_begin = R_cut
	R_end =  R_cut+R_width
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles)[0]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	t = (RijRij2 - R_begin)/R_width
	Cut_step1  = tf.where(tf.greater(t, 0.0), -t*t*(2.0*t-3.0), tf.zeros_like(t))
	Cut = tf.where(tf.greater(t, 1.0), tf.ones_like(t), Cut_step1)

	ZAll = AllDoublesSet(Zs, prec=tf.int64)
	ZPairs1 = tf.slice(ZAll,[0,0,0,1],[nmol,natom,natom,1])
	ZPairs2 = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1])
	Ri=tf.gather_nd(ZPairs1, Radpair)
	Rl=tf.gather_nd(ZPairs2, Radpair)
	ElemIndex_i = tf.slice(tf.where(tf.equal(Ri, tf.reshape(eles, [1,nele]))),[0,1],[nnz,1])
	ElemIndex_j = tf.slice(tf.where(tf.equal(Rl, tf.reshape(eles, [1,nele]))),[0,1],[nnz,1])

	c6_i=tf.gather_nd(c6, ElemIndex_i)
	c6_j=tf.gather_nd(c6, ElemIndex_j)
	Rvdw_i = tf.gather_nd(R_vdw, ElemIndex_i)
	Rvdw_j = tf.gather_nd(R_vdw, ElemIndex_j)
	Kern = -Cut*tf.sqrt(c6_i*c6_j)/tf.pow(RijRij2,6.0)*1.0/(1.0+6.0*tf.pow(RijRij2/(Rvdw_i+Rvdw_j),-12.0))

	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_vdw = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_vdw

def TFVdwPolyLRWithEle(R, Zs, eles, c6, R_vdw, R_cut, Radpair_E1E2, prec=tf.float64):
	"""
	Tensorflow implementation of short range cutoff sparse-coulomb
	Madelung energy build. Using switch function 1+x^2(2x-3) in http://pubs.acs.org/doi/ipdf/10.1021/ct501131j
	damping function in http://pubs.rsc.org/en/content/articlepdf/2008/cp/b810189b is used.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		c6 : nele. Grimmer C6 coff in a.u.
		R_vdw: nele. Grimmer vdw radius in a.u.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	Radpair = Radpair_E1E2[:,:3]
	R = tf.multiply(R, BOHRPERA)
	R_width = PARAMS["Poly_Width"]*BOHRPERA
	R_begin = R_cut
	R_end =  R_cut+R_width
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles)[0]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	t = (RijRij2 - R_begin)/R_width
	Cut_step1  = tf.where(tf.greater(t, 0.0), -t*t*(2.0*t-3.0), tf.zeros_like(t))
	Cut = tf.where(tf.greater(t, 1.0), tf.ones_like(t), Cut_step1)

	ElemIndex_i = tf.reshape(Radpair_E1E2[:,3],[nnz, 1])
	ElemIndex_j = tf.reshape(Radpair_E1E2[:,4],[nnz, 1])

	c6_i=tf.gather_nd(c6, ElemIndex_i)
	c6_j=tf.gather_nd(c6, ElemIndex_j)
	Rvdw_i = tf.gather_nd(R_vdw, ElemIndex_i)
	Rvdw_j = tf.gather_nd(R_vdw, ElemIndex_j)
	Kern = -Cut*tf.sqrt(c6_i*c6_j)/tf.pow(RijRij2,6.0)*1.0/(1.0+6.0*tf.pow(RijRij2/(Rvdw_i+Rvdw_j),-12.0))

	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_vdw = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_vdw

def PolynomialRangeSepCoulomb(R,Qs,Radpair,SRRc,LRRc,dx):
	"""
	A tensorflow linear scaling implementation of a short-range and long range cutoff
	coulomb kernel. The cutoff functions are polynomials subject to the constraint
	that 1/r is brought to 0 twice-differentiably at SR and LR+dx cutoffs.

	The SR cutoff polynomial is 4th order, and the LR is fifth.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		SRRc: Distance where SR polynomial ends.
		LRRc: Distance where LR polynomial begins.
		dx: Small interval after which the kernel is zero.
	Returns
		A #Mols X MaxNAtom X MaxNAtom matrix of LJ kernel contributions.
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	Ds = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	twooversqrtpi = tf.constant(1.1283791671,dtype=tf.float64)
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	Qij = Qi*Qj
	D2 = Ds*Ds
	D3 = D2*Ds
	D4 = D3*Ds
	D5 = D4*Ds

	asr = -5./(3.*tf.pow(SRRc,4.0))
	dsr = 5./(3.*SRRc)
	csr = 1./(tf.pow(SRRc,5.0))

	x0 = LRRc
	x02 = x0*x0
	x03 = x02*x0
	x04 = x03*x0
	x05 = x04*x0

	dx2 = dx*dx
	dx3 = dx2*dx
	dx4 = dx3*dx
	dx5 = dx4*dx

	alr = -((3.*(dx4+2.*dx3*x0-4.*dx2*x02+10.*dx*x03+20.*x04))/(dx5*x03))
	blr = -((-dx5-9*dx4*x0+8.*dx2*x03-60.0*dx*x04-60.0*x05)/(dx5*x03))
	clr = (3.*(dx3-dx2*x0+10.*x03))/(dx5*x03)
	dlr = -((3.*(dx5+3.*dx4*x0-2.*dx3*x02+dx2*x03+15.*dx*x04+10.*x05))/(dx5*x02))
	elr = (3.*(dx5+dx4*x0-dx3*x02+dx2*x03+4.*dx*x04+2.*x05))/(dx5*x0)
	flr = -((dx2-3.*dx*x0+6.*x02)/(dx5*x03))

	CK = (Qij/Ds)
	SRK = Qij*(asr*D3+csr*D4+dsr)
	LRK = Qij*(alr*D3 + blr*D2 + dlr*Ds + clr*D4 + elr + flr*D5)
	ZK = tf.zeros_like(Ds)

	K0 = tf.where(tf.less_equal(Ds,SRRc),SRK,CK)
	K1 = tf.where(tf.greater_equal(Ds,LRRc),LRK,K0)
	K = tf.where(tf.greater_equal(Ds,LRRc+dx),ZK,K1)

	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	sparse_index = tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, K, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	# Now use the sparse reduce sum trick to scatter this into mols.
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def TFCoulombErfLR(R, Qs, R_cut,  Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of long range cutoff sparse-Erf
	Madelung energy build.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	R_width = PARAMS["Erf_Width"]*BOHRPERA
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	# Generate LR cutoff Matrix
	Cut = (1.0 + tf.erf((RijRij2 - R_cut)/R_width))*0.5
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2*Cut
	# Scatter Back
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee


def TFCoulombErfSRDSFLR(R, Qs, R_srcut, R_lrcut, Radpair, alpha, prec=tf.float64):
	"""
	A tensorflow linear scaling implementation of the Damped Shifted Electrostatic Force with short range cutoff
	http://aip.scitation.org.proxy.library.nd.edu/doi/pdf/10.1063/1.2206581
	Batched over molecules.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_srcut: Short Range Erf Cutoff
		R_lrcut: Long Range DSF Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		alpha: DSF alpha parameter (~0.2)
	Returns
		Energy of  Mols
	"""
	alpha = alpha/BOHRPERA
	R_width = PARAMS["Erf_Width"]*BOHRPERA
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Cut = (1.0 + tf.erf((RijRij2 - R_srcut)/R_width))*0.5

	twooversqrtpi = tf.constant(1.1283791671,dtype=tf.float64)
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Gather desired LJ parameters.
	Qij = Qi*Qj
	# This is Dan's Equation (18)
	XX = alpha*R_lrcut
	ZZ = tf.erfc(XX)/R_lrcut
	YY = twooversqrtpi*alpha*tf.exp(-XX*XX)/R_lrcut
	K = Qij*(tf.erfc(alpha*RijRij2)/RijRij2 - ZZ + (RijRij2-R_lrcut)*(ZZ/R_lrcut+YY))*Cut
	K = tf.where(tf.is_nan(K),tf.zeros_like(K),K)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	sparse_index = tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, K, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	# Now use the sparse reduce sum trick to scatter this into mols.
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)


def TFSymRSet_Linear(R, Zs, eles_, SFPs_, eta, R_cut, Radpair, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	ZAll = AllDoublesSet(Zs, prec=tf.int64)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1])
	Rl=tf.gather_nd(ZPairs, Radpair)
	ElemIndex = tf.slice(tf.where(tf.equal(Rl, tf.reshape(eles_,[1,nele]))),[0,1],[nnz,1])
	GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds2,[0,0],[nnz,2]),tf.slice(GoodInds2,[0,3],[nnz,1]),tf.slice(GoodInds2,[0,2],[nnz,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])
	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p2_2 = tf.reshape(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[nr,1]),[1,nr,1])
	p4_2 = tf.tile(p2_2,[nnz,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(RadpairEle,[0,0],[nnz,2]),tf.slice(RadpairEle,[0,3],[nnz,1]),tf.slice(RadpairEle,[0,2],[nnz,1])],axis=-1)
	#mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])

	to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
#	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
#	p2_2 = tf.reshape(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[nr,1]),[1,nr,1])
#	p4_2 = tf.tile(p2_2,[nnz,1,1]) # should be nnz X nsym
#	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
#	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
#	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)

def TFSymRSet_Linear_WithEle_Release(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, mil_j, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1
	to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol, tf.cast(natom, tf.int32), nele, tf.cast(j_max, tf.int32), nsym], dtype=tf.int64))
	#mil_j = tf.concat([tf.slice(RadpairEle,[0,0],[nnz,2]),tf.slice(RadpairEle,[0,3],[nnz,1]),tf.slice(RadpairEle,[0,2],[nnz,1])],axis=-1)
	#to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymRSet_Linear_WithEle_Channel(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, mil_j, channels, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1
	to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol, tf.cast(natom, tf.int32), nele, tf.cast(j_max, tf.int32), nsym], dtype=tf.int64))

	tomul_channels = tf.cast(tf.reshape(channels,[1,1,nele,1]), dtype=tf.float64)
	to_reduce3 = tf.reduce_sum(to_reduce2, axis=3)
	to_reduce4 = tomul_channels*to_reduce3



	#mil_j = tf.concat([tf.slice(RadpairEle,[0,0],[nnz,2]),tf.slice(RadpairEle,[0,3],[nnz,1]),tf.slice(RadpairEle,[0,2],[nnz,1])],axis=-1)
	#to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	return tf.reduce_sum(to_reduce4, axis=2)


def TFSymRSet_Linear_WithEle_Channel2(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, mil_j, channels, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	num_ele, num_dim = eles_.get_shape().as_list()
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)
	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1

	tomul = tf.reshape(tf.gather_nd(channels, tf.reshape(mil_j[:,2],[-1,1])),[-1,1])
	Gm2_mult = tf.cast(tomul, dtype=tf.float64)*Gm2

	to_reduce2  = tf.zeros([nmol, tf.cast(natom, tf.int32), tf.cast(j_max, tf.int32), nsym], dtype=tf.float64)
	for e in range(num_ele):
		mask_tensor = tf.equal(mil_j[:,2],e)
		mil_j_mask = tf.boolean_mask(mil_j, mask_tensor)
		mi_j_mask =  tf.concat([mil_j_mask[:,:2],tf.reshape(mil_j_mask[:,3],[-1,1])], axis=-1)
		Gm2_mult_mask = tf.boolean_mask(Gm2_mult, mask_tensor)
		tmp = tf.scatter_nd(mi_j_mask, Gm2_mult_mask, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(j_max, tf.int32), nr], dtype=tf.int64))
		to_reduce2 = to_reduce2 + tmp
	return tf.reduce_sum(to_reduce2, axis=2)

def TFSymRSet_Linear_WithEle_Channel3(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, mil_j, channels, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	num_ele, num_dim = eles_.get_shape().as_list()
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)
	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1

	tomul = tf.reshape(tf.gather_nd(channels, tf.reshape(mil_j[:,2],[-1,1])),[nnz, -1])

	Gm2_mult = tf.reshape(tf.expand_dims(tomul, 1)*tf.expand_dims(Gm2, 2),[nnz, -1])


	mi_j =  tf.concat([mil_j[:,:2],tf.reshape(mil_j[:,3],[-1,1])], axis=-1)
	to_reduce2 = tf.scatter_nd(mi_j, Gm2_mult, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(j_max, tf.int32), nr*(tf.shape(channels)[1])], dtype=tf.int64))

	return tf.reduce_sum(to_reduce2, axis=2)

def TFSymRSet_Linear_WithEle_ChannelHyb(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, mil_j, channels, Hyb, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	num_ele, num_dim = eles_.get_shape().as_list()
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)
	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1

	tomul = tf.reshape(tf.gather_nd(channels, tf.reshape(mil_j[:,2],[-1,1])),[nnz, -1])

	mj_inds = tf.concat([RadpairEle[:,0], RadpairEle[:,2]],axis=-1)
	hyb_channel  = tf.gather_nd(Hyb, tf.reshape(mj_inds,[-1,2]))
	tomul_with_hyb = tf.concat([tomul, hyb_channel],axis=-1)

	Gm2_mult = tf.reshape(tf.expand_dims(tomul_with_hyb, 1)*tf.expand_dims(Gm2, 2),[nnz, -1])


	mi_j =  tf.concat([mil_j[:,:2],tf.reshape(mil_j[:,3],[-1,1])], axis=-1)
	to_reduce2 = tf.scatter_nd(mi_j, Gm2_mult, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(j_max, tf.int32), nr*(tf.shape(channels)[1]+tf.shape(Hyb)[2])], dtype=tf.int64))

	return tf.reduce_sum(to_reduce2, axis=2)
def TFSymSet_Hybrization(R, Zs, eles_, eleps_,  R_cut, RadpairEle, mil_j, AngtriEle, mil_jk, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	num_ele, num_dim = eles_.get_shape().as_list()
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	weight = (1.0-tf.tanh((RijRij - R_cut)*2.0*3.1415))/2.0
	mi_j =  tf.concat([mil_j[:,:2],tf.reshape(mil_j[:,3],[-1,1])], axis=-1)
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1

	scatter_weight = tf.scatter_nd(mi_j, weight, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(j_max, tf.int32)], dtype=tf.int64))
	coord = tf.reduce_sum(scatter_weight, axis=2)


	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	num_elep, num_dim = eleps_.get_shape().as_list()
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)

	weighta = (1.0-tf.tanh((RijRij2 - R_cut)*2.0*3.1415))/2.0
	weightb = (1.0-tf.tanh((RikRik2 - R_cut)*2.0*3.1415))/2.0
	weightab = weighta*weightb

	mi_jk2 =  tf.concat([mil_jk[:,:2],tf.reshape(mil_jk[:,3],[-1,1])], axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk,[0,3], [nnzt, 1])) + 1
	scatter_weightab = tf.scatter_nd(mi_jk2, weightab, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(jk_max, tf.int32)], dtype=tf.int64))
	scatter_angle = tf.scatter_nd(mi_jk2, Thetaijk, tf.cast([nmol, tf.cast(natom, tf.int32), tf.cast(jk_max, tf.int32)], dtype=tf.int64))
	weightab_sum = tf.reduce_sum(scatter_weightab, axis=2)
	scatter_angle_w= scatter_angle*scatter_weightab
	scatter_angle_wsum = tf.reduce_sum(scatter_angle_w, axis=2)
	avg_angle = scatter_angle_wsum/(weightab_sum+infinitesimal)

	tmp = tf.expand_dims(avg_angle,2) - scatter_angle
	std = tf.reduce_sum(tmp*tmp*scatter_weightab, axis=2)/(weightab_sum+infinitesimal)
	output = tf.stack([coord, avg_angle, std], axis=2)
	#output_clean = tf.where(tf.is_nan(output), tf.zeros_like(output, dtype=prec), output)
	return output

def TFSymRSet_Linear_WithElePeriodic(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, mil_j, nreal, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1
	#mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])
	to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol, tf.cast(nreal, tf.int32), nele, tf.cast(j_max, tf.int32), nsym], dtype=tf.int64))
	return tf.reduce_sum(to_reduce2, axis=3)

def TFSymRSet_Linear_Qs(R, Zs, eles_, SFPs_, eta, R_cut, Radpair, Qs, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		Qs: charge of each atom. nmol X maxnatom
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	ZAll = AllDoublesSet(Zs, prec=tf.int64)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1])
	Rl=tf.gather_nd(ZPairs, Radpair)
	ElemIndex = tf.slice(tf.where(tf.equal(Rl, tf.reshape(eles_,[1,nele]))),[0,1],[nnz,1])
	GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	Qit = tf.tile(tf.reshape(Qi,[nnz,1]),[1, nr])
	Qjt = tf.tile(tf.reshape(Qj,[nnz,1]),[1, nr])

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t*Qit*Qjt,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr

	## Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds2,[0,0],[nnz,2]),tf.slice(GoodInds2,[0,3],[nnz,1]),tf.slice(GoodInds2,[0,2],[nnz,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])
	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p2_2 = tf.reshape(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[nr,1]),[1,nr,1])
	p4_2 = tf.tile(p2_2,[nnz,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(tf.reduce_sum(to_reduce2, axis=3), axis=2)



def TFSymRSet_Linear_Qs_Periodic(R, Zs, eles_, SFPs_, eta, R_cut, Radpair, Qs, mil_j, nreal, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		Qs: charge of each atom. nmol X maxnatom
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	Qit = tf.tile(tf.reshape(Qi,[nnz,1]),[1, nr])
	Qjt = tf.tile(tf.reshape(Qj,[nnz,1]),[1, nr])

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t*Qit*Qjt,[nnz, nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1
	to_reduce2 = tf.scatter_nd(mil_j, Gm, tf.cast([nmol, tf.cast(nreal, tf.int32), nele, tf.cast(j_max, tf.int32), nsym], dtype=tf.int64))
	return tf.reduce_sum(tf.reduce_sum(to_reduce2, axis=3), axis=2)


def TFSymSet(R, Zs, eles_, SFPsR_, Rr_cut, eleps_, SFPsA_, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet(R, Zs, eles_, SFPsR_, Rr_cut),[nmol, natom, -1])
	GMA = tf.reshape(TFSymASet(R, Zs, eleps_, SFPsA_, Ra_cut),[nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	return GM

def TFSymSet_Scattered(R, Zs, eles_, SFPsR_, Rr_cut, eleps_, SFPsA_, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet(R, Zs, eles_, SFPsR_, Rr_cut),[nmol, natom, -1])
	GMA = tf.reshape(TFSymASet(R, Zs, eleps_, SFPsA_, Ra_cut),[nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask = AllSinglesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	IndexList = []
	SymList=[]
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		SymList.append(tf.gather_nd(GM, GatherList[-1]))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		IndexList.append(tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle]))
	return SymList, IndexList

def TFSymSet_Scattered_Update(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Update(R, Zs, eles_, SFPsR_, Rr_cut), [nmol, natom, -1])
	GMA = tf.reshape(TFSymASet_Update(R, Zs, eleps_, SFPsA_, Ra_cut), [nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask = AllSinglesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	IndexList = []
	SymList=[]
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		SymList.append(tf.gather_nd(GM, GatherList[-1]))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		IndexList.append(tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle]))
	return SymList, IndexList


def TFSymSet_Scattered_Update2(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Update2(R, Zs, eles_, SFPsR_, eta, Rr_cut), [nmol, natom, -1])
	GMA = tf.reshape(TFSymASet_Update2(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut), [nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	IndexList = []
	SymList=[]
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		SymList.append(tf.gather_nd(GM, GatherList[-1]))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		IndexList.append(tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle]))
	return SymList, IndexList

def TFSymSet_Scattered_Update_Scatter(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	This also selects out which of the atoms will contribute to the BP energy on the
	basis of whether the atomic number is treated in the 'element types to do list.'
	according to kun? (Trusted source?)

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Update2(R, Zs, eles_, SFPsR_, eta, Rr_cut), [nmol, natom, -1])
	GMA = tf.reshape(TFSymASet_Update2(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut), [nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64)
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, Radp, Angt):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear(R, Zs, eles_, SFPsR_, eta, Rr_cut, Radp),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  Angt), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList


def TFSymSet_Scattered_Linear_WithEle(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_jk):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk),[nmol, natom,-1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_Release(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear_WithEle_Release(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk),[nmol, natom,-1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_Channel(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]

	channel_eles = tf.reshape(eles_, [nele])
	channel_eleps = tf.reshape((eleps_[:,0] +eleps_[:,1])/(eleps_[:,0]*eleps_[:,1]),[nelep])

	GMR = tf.reshape(TFSymRSet_Linear_WithEle_Channel(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j, channel_eles),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithEle_Channel(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk, channel_eleps),[nmol, natom,-1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_Channel2(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]

	channel_eles = tf.reshape(eles_, [nele])
	channel_eleps = tf.reshape((eleps_[:,0] +eleps_[:,1])/(eleps_[:,0]*eleps_[:,1]),[nelep])

	GMR = tf.reshape(TFSymRSet_Linear_WithEle_Channel2(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j, channel_eles),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithEle_Channel2(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk, channel_eleps),[nmol, natom,-1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_Channel3(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk, channel_eles, channel_eleps):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]

	#channel_eles = tf.reshape(eles_, [nele])
	#channel_eleps = tf.reshape((eleps_[:,0] +eleps_[:,1])/(eleps_[:,0]*eleps_[:,1]),[nelep])

	GMA = tf.reshape(TFSymASet_Linear_WithEle_Channel3(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk, channel_eleps),[nmol, natom,-1], name="FinishGMA")
	GMR = tf.reshape(TFSymRSet_Linear_WithEle_Channel3(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j, channel_eles),[nmol, natom,-1], name="FinishGMR")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAngHyb")
	#GM = tf.identity(GMR)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_ChannelHyb(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk, channel_eles, channel_eleps):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]

	#channel_eles = tf.reshape(eles_, [nele])
	#channel_eleps = tf.reshape((eleps_[:,0] +eleps_[:,1])/(eleps_[:,0]*eleps_[:,1]),[nelep])

	GMH = tf.reshape(TFSymSet_Hybrization(R, Zs, eles_, eleps_,  2.0, RadpEle, mil_j, AngtEle, mil_jk), [nmol, natom,-1], name="FinishGMH")
	GMA = tf.reshape(TFSymASet_Linear_WithEle_ChannelHyb(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk, channel_eleps, GMH),[nmol, natom,-1], name="FinishGMA")
	GMR = tf.reshape(TFSymRSet_Linear_WithEle_ChannelHyb(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j, channel_eles, GMH),[nmol, natom,-1], name="FinishGMR")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAngHyb")
	#GM = tf.identity(GMR)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList


def TFSymSet_Scattered_Linear_WithEle_Channel_Multi(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk, channel_eles, channel_eleps):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	GMA = tf.reshape(TFSymASet_Linear_WithEle_Channel3(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk, channel_eleps),[nmol, natom,-1], name="FinishGMA")
	GMR = tf.reshape(TFSymRSet_Linear_WithEle_Channel3(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j, channel_eles),[nmol, natom,-1], name="FinishGMR")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAngHyb")
	return tf.reshape(GM,[nmol, natom, -1, tf.shape(channel_eles)[1]])


def TFSymSet_Scattered_Linear_WithEle_UsingList(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_jk):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithEle_UsingList(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk),[nmol, natom,-1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_Periodic(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk, nreal):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear_WithElePeriodic(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j, nreal),[nmol, nreal, -1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithElePeriodic(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk, nreal),[nmol, nreal, -1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	Zs_real = Zs[:,:nreal]
	MaskAll = tf.equal(tf.reshape(Zs_real,[nmol,nreal,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(nreal),[1,nreal]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*nreal), [nmol, nreal, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,nreal,1]),[nmol, nreal])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Radius_Scattered_Linear_Qs(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_,  eta,  Radp, Qs):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GM = tf.reshape(TFSymRSet_Linear_Qs(R, Zs, eles_, SFPsR_, eta, Rr_cut, Radp, Qs),[nmol, natom,-1])
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64)
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Radius_Scattered_Linear_Qs_Periodic(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_,  eta,  Radp, Qs, mil_j, nreal):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GM = tf.reshape(TFSymRSet_Linear_Qs_Periodic(R, Zs, eles_, SFPsR_, eta, Rr_cut, Radp, Qs, mil_j, nreal),[nmol, nreal,-1])
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	Zs_real = Zs[:,:nreal]
	MaskAll = tf.equal(tf.reshape(Zs_real,[nmol,nreal,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(nreal),[1,nreal]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*nreal), [nmol, nreal, 1]), dtype=tf.int64)
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,nreal,1]),[nmol, nreal])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def NNInterface(R, Zs, eles_, GM):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom  tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		GM: Unscattered ANI1 sym Func: nmol X natom X nele X Dim


	Returns:
		List of ANI SymFunc of each atom by element type.
		List of Mol index of each atom by element type.
	"""
	nele = tf.shape(eles_)[0]
	num_ele, num_dim = eles_.get_shape().as_list()
	R_shp = tf.shape(R)
	nmol = R_shp[0]
	natom = R_shp[1]
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask = AllSinglesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	IndexList = []
	SymList=[]
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		SymList.append(tf.gather_nd(GM, GatherList[-1]))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		IndexList.append(tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle]))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_Channel_Multitmp(xyzs, Zs, neighbors, elements, element_pairs, element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		xyzs: nmol X maxnatom X 3 tensor of coordinates.
		Zs: nmol X max_n_atom tensor of atomic numbers.
		elements: a neles X 1 tensor of elements present in the data.
		element_pairs: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		element_codes: n_elements x 4 tensor of codes for embedding elements
		radial_gauss: A symmetry function parameter of radius part
		radial_cutoff: Radial Cutoff of radius part
		angular_gauss: A symmetry function parameter of angular part
		angular_cutoff: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	dxyzs, pair_Zs = sparse_coords(xyzs, Zs, neighbors)
	radial_embed = tf_radial_sym_func(dxyzs, pair_Zs, element_codes, radial_gauss, radial_cutoff, eta)
	dtxyzs, triples_Zs = sparse_triples(xyzs, Zs, neighbors)
	return TFSymASet_Linear_WithEle_Channel3tmp(dtxyzs, triples_Zs, element_codes, angular_gauss, thetas, angular_cutoff, zeta, eta)
	# GMA = tf.reshape(TFSymASet_Linear_WithEle_Channel3tmp(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk, channel_eleps),[nmol, natom,-1], name="FinishGMA")
	# GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAngHyb")
	return tf.reshape(GM,[nmol, natom, -1, tf.shape(channel_eles)[1]])

def tf_radial_sym_func(dxyzs, pair_Zs, element_codes, radial_gauss, radial_cutoff, eta):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		dxyzs: n_case X max_neighbors X 3 tensor of coordinates.
		Zs: n_case X max_neighbors tensor of atomic numbers.
		element_codes: n_elements x 4 tensor of codes for embedding element type
		radial_gauss: n_gauss tensor of radial gaussian centers
		radial_cutoff: radial cutoff distance
		eta: radial gaussian width parameter
	Returns:
		radial_embed: n_case X 4 x n_gauss tensor of atoms embeded into central atoms environment
	"""
	dist_tensor = tf.norm(dxyzs+1.e-16, axis=-1)
	exponent = tf.square(tf.expand_dims(dist_tensor, axis=-1) - radial_gauss)
	exponent *= -1.0 * eta
	gauss = tf.exp(exponent)
	cutoff = 0.5 * (tf.cos(np.pi * dist_tensor / radial_cutoff) + 1.0)
	pair_codes = tf.gather(element_codes, pair_Zs)
	radial_embed = tf.expand_dims(gauss, axis=-2) * tf.expand_dims(tf.expand_dims(cutoff, axis=-1) * pair_codes, axis=-1)
	return tf.reduce_sum(radial_embed, axis=1)

def TFSymASet_Linear_WithEle_Channel3tmp(dtxyzs, triples_Zs, element_codes, angular_gauss, thetas, angular_cutoff, zeta, eta):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	dist_jk_tensor = tf.norm(dtxyzs+1.e-16, axis=-1)
	dij_dik = tf.reduce_prod(dist_jk_tensor, axis=-1)
	ij_dot_ik = tf.reduce_sum(dtxyzs[...,0,:] * dtxyzs[...,1,:], axis=-1)
	cos_angle = ij_dot_ik / dij_dik
	#cos_angle = tf.where(tf.greater_equal(cos_angle, 1.0), tf.ones_like(cos_angle, dtype=eval(PARAMS["tf_prec"])) - 1.e-16, cos_angle)
	#cos_angle = tf.where(tf.less_equal(cos_angle, -1.0), -1.0 * tf.ones_like(cos_angle, dtype=eval(PARAMS["tf_prec"])) - 1.e-16, cos_angle)
	# sin_angle = tf.norm(tf.cross(dtxyzs[...,0,:], dtxyzs[...,1,:]), axis=-1) / dij_dik
	
	theta_ijk = tf.atan2(sin_angle, cos_angle)
	return tf.reduce_max(sin_angle)

	# theta_ijk = tf.acos(cos_angle)
	# return theta_ijk
	dtheta = tf.expand_dims(theta_ijk, axis=-1) - thetas
	cos_factor = tf.cos(dtheta)
	exponent = tf.expand_dims(tf.reduce_sum(dist_jk_tensor, axis=-1) / 2.0, axis=-1) - angular_gauss
	dist_factor = tf.exp(-eta * tf.square(exponent))
	cutoff = tf.reduce_prod(0.5 * (tf.cos(np.pi * dist_jk_tensor / angular_cutoff) + 1.0), axis=-1)
	angular_embed = tf.expand_dims(tf.pow(1.0 + cos_factor, zeta), axis=-1) * tf.expand_dims(dist_factor, axis=-2)
	angular_embed *= tf.expand_dims(tf.expand_dims(cutoff, axis=-1), axis=-1)
	angular_embed = tf.expand_dims(angular_embed, axis=-3) * tf.expand_dims(tf.expand_dims(tf.reduce_prod(tf.gather(element_codes, triples_Zs), axis=-2), axis=-1), axis=-1)
	angular_embed = tf.pow(2.0, 1.0 - zeta) * tf.reduce_sum(angular_embed, axis=(1,2))
	return angular_embed

def sparse_coords(xyzs, Zs, pairs):
	padding_mask = tf.where(tf.not_equal(Zs, 0))
	central_atom_coords = tf.gather_nd(xyzs, padding_mask)
	pairs = tf.gather_nd(pairs, padding_mask)
	padded_pairs = tf.equal(pairs, -1)
	tmp_pairs = tf.where(padded_pairs, tf.zeros_like(pairs), pairs)
	gather_pairs = tf.stack([tf.cast(tf.tile(padding_mask[:,:1], [1, tf.shape(pairs)[1]]), tf.int32), tmp_pairs], axis=-1)
	pair_coords = tf.gather_nd(xyzs, gather_pairs)
	dxyzs = tf.expand_dims(central_atom_coords, axis=1) - pair_coords
	pair_mask = tf.where(padded_pairs, tf.zeros_like(pairs), tf.ones_like(pairs))
	dxyzs *= tf.cast(tf.expand_dims(pair_mask, axis=-1), eval(PARAMS["tf_prec"]))
	pair_Zs = tf.gather_nd(Zs, gather_pairs)
	pair_Zs *= pair_mask
	return dxyzs, pair_Zs

def sparse_triples(xyzs, Zs, pairs):
	padding_mask = tf.where(tf.not_equal(Zs, 0))
	central_atom_coords = tf.gather_nd(xyzs, padding_mask)
	pairs = tf.gather_nd(pairs, padding_mask)
	idx_sorted_pairs, _ = tf.nn.top_k(pairs, tf.shape(pairs)[-1], True)
	tiled_sorted_pairs1 = tf.tile(tf.expand_dims(idx_sorted_pairs, axis=-1), [1, 1, tf.shape(idx_sorted_pairs)[-1]])
	tiled_sorted_pairs2 = tf.tile(tf.expand_dims(idx_sorted_pairs, axis=-2), [1, tf.shape(idx_sorted_pairs)[-1], 1])
	all_triples = tf.stack([tiled_sorted_pairs1, tiled_sorted_pairs2], axis=-1)
	triples_mask = tf.logical_and(tf.not_equal(all_triples[...,0], -1), tf.not_equal(all_triples[...,1], -1))
	triples_mask = tf.logical_and(tf.greater(all_triples[...,0], all_triples[...,1]), triples_mask)
	triples_mask = tf.tile(tf.expand_dims(triples_mask, axis=-1), [1, 1, 1, 2])
	all_triples = tf.where(tf.logical_not(triples_mask), -1 * tf.ones_like(all_triples), all_triples)
	tmp_triples = tf.where(tf.logical_not(triples_mask), tf.zeros_like(all_triples), all_triples)
	mol_idx = tf.tile(tf.reshape(padding_mask[:,0], [-1, 1, 1, 1]), [1, tf.shape(idx_sorted_pairs)[-1], tf.shape(idx_sorted_pairs)[-1], 2])
	gather_triples = tf.stack([tf.cast(mol_idx, tf.int32), tmp_triples], axis=-1)
	triples_coords = tf.gather_nd(xyzs, gather_triples)
	dxyzs = tf.reshape(central_atom_coords, [tf.shape(padding_mask)[0], 1, 1, 1, 3]) - triples_coords
	triples_mask = tf.where(triples_mask, tf.ones_like(all_triples), tf.zeros_like(all_triples))
	dxyzs *= tf.cast(tf.expand_dims(triples_mask, axis=-1), eval(PARAMS["tf_prec"]))
	triples_Zs = tf.gather_nd(Zs, gather_triples)
	triples_Zs *= triples_mask
	return dxyzs, triples_Zs
