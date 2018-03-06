"""
Routines which do elementary math in Tensorflow which is for whatever reason missing in the Tensorflow API.
"""
import tensorflow as tf

def TFMatrixPower(mat_,exp_):
	"""
	General Matrix Power in Tensorflow.
	This is NOT differentiable as of 1.2.
	tf.matrix_inverse and tf.matrix_determinant are though.
	"""
	s,u,v = tf.svd(mat_,full_matrices=True,compute_uv=True)
	return tf.transpose(tf.matmul(u,tf.matmul(tf.diag(tf.pow(s,exp_)),tf.transpose(v))))

def TFMatrixSqrt(mat_):
	"""
	Use Denman-Beavers iteration to compute a
	Matrix Square root differentiably.
	"""
	cond = lambda i,y,z: i<10
	body = lambda i,y,z: [i+1,0.5*(y+tf.matrix_inverse(z)),0.5*(z+tf.matrix_inverse(y))]
	initial = (0,a,tf.eye(tf.shape(a)[0]))
	I,Y,Z = tf.while_loop(cond,body,initial)
	return Y,Z

def TFDistance(A):
	"""
	Compute a distance matrix of A, a coordinate matrix
	Using the factorization:
	Dij = <i|i> - 2<i|j> + <j,j>
	Args:
		A: a Nx3 matrix
	Returns:
		D: a NxN matrix
	"""
	r = tf.reduce_sum(A*A, 1)
	r = tf.reshape(r, [-1, 1]) # For the later broadcast.
	# Tensorflow can only reverse mode grad the sqrt if all these elements
	# are nonzero
	D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
	return tf.sqrt(tf.clip_by_value(D,1e-36,1e36))

def TFDistances(r_):
	"""
	Returns distance matrices batched over mols
	Args:
		r_: Nmol X MaxNAtom X 3 coordinate tensor
	Returns
		D: Nmol X MaxNAtom X MaxNAtom Distance tensor.
	"""
	rm = tf.einsum('ijk,ijk->ij',r_,r_) # Mols , Rsq.
	rshp = tf.shape(rm)
	rmt = tf.tile(rm, [1,rshp[1]])
	rmt = tf.reshape(rmt,[rshp[0],rshp[1],rshp[1]])
	rmtt = tf.transpose(rmt,perm=[0,2,1])
	# Tensorflow can only reverse mode grad of sqrt if all these elements
	# are nonzero
	D = rmt - 2*tf.einsum('ijk,ilk->ijl',r_,r_) + rmtt
	return tf.sqrt(tf.clip_by_value(D,1e-36,1e36))

def TFDistanceLinear(B,NZP):
	"""
	Compute a distance vector of B, a coordinate matrix
	Args:
		B: a Nx3 matrix
		NZP: a (nonzero pairs X 2) index matrix.
	Returns:
		D: a NZP X 1 tensor of distances.
	"""
	Ii = tf.slice(NZP,[0,0],[-1,1])
	Ij = tf.slice(NZP,[0,1],[-1,1])
	Ri = tf.gather_nd(B,Ii)
	Rj = tf.gather_nd(B,Ij)
	A = Ri - Rj
	return tf.sqrt(tf.clip_by_value(tf.reduce_sum(A*A, 1),1e-36,1e36))

def TFDistancesLinear(B,NZP):
	"""
	Returns distance vector batched over mols
	With these sparse versions I think the mol dimension should be eliminated.

	Args:
		r_: Nmol X MaxNAtom X 3 coordinate tensor
		NZP: a ( nonzero pairs X 3) index matrix. (mol, i, j)
	Returns
		D: nonzero pairs  Distance vector. (Dij)
		The calling routine has to scatter back into mol dimension using NZP if desired.
	"""
	Ii = tf.slice(NZP,[0,0],[-1,2])
	Ij = tf.concat([tf.slice(NZP,[0,0],[-1,1]),tf.slice(NZP,[0,2],[-1,1])],1)
	Ri = tf.gather_nd(B,Ii)
	Rj = tf.gather_nd(B,Ij)
	A = Ri - Rj
	D = tf.sqrt(tf.clip_by_value(tf.reduce_sum(A*A, 1),1e-36,1e36))
	return D

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

def TF_RandomRotationBatch(sz_,max_dist=1.0):
	"""
	Returns a batch of uniform rotation matrices,
	and the angles of each. Finds random unit vector
	and then random angle around it.

	Args:
		sz_: number of rotation matrices
		max_dist: maximum rotation in units of 2*Pi
	"""
	Pi = 3.14159265359
	thetas = tf.acos(2.0*tf.random_uniform(sz_,dtype=tf.float64)-1)
	phis = tf.random_uniform(sz_,dtype=tf.float64)*2*Pi
	axes = tf.zeros(shape=sz_+[3])
	axes = tf.stack([tf.sin(thetas)*tf.cos(phis), tf.sin(thetas)*tf.sin(phis), tf.cos(thetas)],axis=-1)
	psis = tf.random_uniform(sz_,dtype=tf.float64)*2*Pi*max_dist
	ct = tf.cos(psis)
	st = tf.sin(psis)
	omct = 1.0-ct
	matrices = tf.reshape(tf.stack([ct+axes[:,0]*axes[:,0]*omct,
	axes[:,0]*axes[:,1]*omct - axes[:,2]*st,
	axes[:,0]*axes[:,2]*omct + axes[:,1]*st,
	axes[:,1]*axes[:,0]*omct + axes[:,2]*st,
	ct+axes[:,1]*axes[:,1]*omct,
	axes[:,1]*axes[:,2]*omct - axes[:,0]*st,
	axes[:,2]*axes[:,0]*omct - axes[:,1]*st,
	axes[:,2]*axes[:,1]*omct + axes[:,0]*st,
	ct + axes[:,2]*axes[:,2]*omct],axis = -1),sz_+[3,3])
	return matrices, tf.stack([thetas,phis,psis],axis=-1)
