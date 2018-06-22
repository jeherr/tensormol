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
from ..Math.TFMath import *
from tensorflow.python.client import timeline
import numpy as np
from tensorflow.python.framework import function
if (HAS_TF):
	import tensorflow as tf

def tf_pairs_list(xyzs, Zs, r_cutoff, element_pairs):
	dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	dist_tensor = tf.norm(dxyzs,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(dist_tensor, r_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
	permutation_identity_mask = tf.where(tf.less(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, permutation_identity_mask)), tf.int32)
	pair_distances = tf.expand_dims(tf.gather_nd(dist_tensor, pair_indices), axis=1)
	pair_elements = tf.stack([tf.gather_nd(Zs, pair_indices[:,0:2]), tf.gather_nd(Zs, pair_indices[:,0:3:2])], axis=-1)
	element_pair_mask = tf.cast(tf.where(tf.logical_or(tf.reduce_all(tf.equal(tf.expand_dims(pair_elements, axis=1), tf.expand_dims(element_pairs, axis=0)), axis=2),
						tf.reduce_all(tf.equal(tf.expand_dims(pair_elements, axis=1), tf.expand_dims(element_pairs[:,::-1], axis=0)), axis=2))), tf.int32)
	num_element_pairs = element_pairs.get_shape().as_list()[0]
	element_pair_distances = tf.dynamic_partition(pair_distances, element_pair_mask[:,1], num_element_pairs)
	mol_indices = tf.dynamic_partition(pair_indices[:,0], element_pair_mask[:,1], num_element_pairs)
	return element_pair_distances, mol_indices

def tf_triples_list(xyzs, Zs, r_cutoff, element_triples):
	num_mols = Zs.get_shape().as_list()[0]
	dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	dist_tensor = tf.norm(dxyzs,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(dist_tensor, r_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
	permutation_identity_mask = tf.where(tf.less(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, permutation_identity_mask)), tf.int32)
	mol_pair_indices = tf.dynamic_partition(pair_indices, pair_indices[:,0], num_mols)
	mol_triples_indices = []
	tmp = []
	for i in xrange(num_mols):
		mol_common_atom_indices = tf.where(tf.reduce_all(tf.equal(tf.expand_dims(mol_pair_indices[i][:,0:2], axis=0), tf.expand_dims(mol_pair_indices[i][:,0:2], axis=1)), axis=2))
		permutation_pairs_mask = tf.where(tf.less(mol_common_atom_indices[:,0], mol_common_atom_indices[:,1]))
		mol_common_atom_indices = tf.squeeze(tf.gather(mol_common_atom_indices, permutation_pairs_mask), axis=1)
		tmp.append(mol_common_atom_indices)
		mol_triples_indices.append(tf.concat([tf.gather(mol_pair_indices[i], mol_common_atom_indices[:,0]), tf.expand_dims(tf.gather(mol_pair_indices[i], mol_common_atom_indices[:,1])[:,2], axis=1)], axis=1))
	triples_indices = tf.concat(mol_triples_indices, axis=0)
	triples_distances = tf.stack([tf.gather_nd(dist_tensor, triples_indices[:,:3]),
						tf.gather_nd(dist_tensor, tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1)),
						tf.gather_nd(dist_tensor, tf.concat([triples_indices[:,0:1], triples_indices[:,2:]], axis=1))], axis=-1)
	cos_thetas = tf.stack([(tf.square(triples_distances[:,0]) + tf.square(triples_distances[:,1]) - tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,0] * triples_distances[:,1]),
						(tf.square(triples_distances[:,0]) - tf.square(triples_distances[:,1]) + tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,0] * triples_distances[:,2]),
						(-tf.square(triples_distances[:,0]) + tf.square(triples_distances[:,1]) + tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,1] * triples_distances[:,2])], axis=-1)
	cos_thetas = tf.where(tf.greater_equal(cos_thetas, 1.0), tf.ones_like(cos_thetas) * (1.0 - 1.0e-24), cos_thetas)
	cos_thetas = tf.where(tf.less_equal(cos_thetas, -1.0), -1.0 * tf.ones_like(cos_thetas) * (1.0 - 1.0e-24), cos_thetas)
	triples_angles = tf.acos(cos_thetas)
	triples_distances_angles = tf.concat([triples_distances, triples_angles], axis=1)
	triples_elements = tf.stack([tf.gather_nd(Zs, triples_indices[:,0:2]), tf.gather_nd(Zs, triples_indices[:,0:3:2]), tf.gather_nd(Zs, triples_indices[:,0:4:3])], axis=-1)
	sorted_triples_elements, _ = tf.nn.top_k(triples_elements, k=3)
	element_triples_mask = tf.cast(tf.where(tf.reduce_all(tf.equal(tf.expand_dims(sorted_triples_elements, axis=1), tf.expand_dims(element_triples, axis=0)), axis=2)), tf.int32)
	num_element_triples = element_triples.get_shape().as_list()[0]
	element_triples_distances_angles = tf.dynamic_partition(triples_distances_angles, element_triples_mask[:,1], num_element_triples)
	mol_indices = tf.dynamic_partition(triples_indices[:,0], element_triples_mask[:,1], num_element_triples)
	return element_triples_distances_angles, mol_indices


def matrix_power(matrix, power):
	"""
	Raise a Hermitian Matrix to a possibly fractional power.

	Args:
		matrix (tf.float): Diagonalizable matrix
		power (tf.float): power to raise the matrix to

	Returns:
		matrix_to_power (tf.float): matrix raised to the power

	Note:
		As of tensorflow v1.3, tf.svd() does not have gradients implimented
	"""
	s, U, V = tf.svd(matrix)
	s = tf.maximum(s, tf.pow(10.0, -14.0))
	return tf.matmul(U, tf.matmul(tf.diag(tf.pow(s, power)), tf.transpose(V)))

def matrix_power2(matrix, power):
	"""
	Raises a matrix to a possibly fractional power

	Args:
		matrix (tf.float): Diagonalizable matrix
		power (tf.float): power to raise the matrix to

	Returns:
		matrix_to_power (tf.float): matrix raised to the power
	"""
	matrix_eigenvals, matrix_eigenvecs = tf.self_adjoint_eig(matrix)
	matrix_to_power = tf.matmul(matrix_eigenvecs, tf.matmul(tf.matrix_diag(tf.pow(matrix_eigenvals, power)), tf.transpose(matrix_eigenvecs)))
	return matrix_to_power

def tf_gauss_overlap(gauss_params):
	r_nought = gauss_params[:,0]
	sigma = gauss_params[:,1]
	scaling_factor = tf.cast(tf.sqrt(np.pi / 2), eval(PARAMS["tf_prec"]))
	exponential_factor = tf.exp(-tf.square(tf.expand_dims(r_nought, axis=0) - tf.expand_dims(r_nought, axis=1))
	/ (2.0 * (tf.square(tf.expand_dims(sigma, axis=0)) + tf.square(tf.expand_dims(sigma, axis=1)))))
	root_inverse_sigma_sum = tf.sqrt((1.0 / tf.expand_dims(tf.square(sigma), axis=0)) + (1.0 / tf.expand_dims(tf.square(sigma), axis=1)))
	erf_numerator = (tf.expand_dims(r_nought, axis=0) * tf.expand_dims(tf.square(sigma), axis=1)
				+ tf.expand_dims(r_nought, axis=1) * tf.expand_dims(tf.square(sigma), axis=0))
	erf_denominator = (tf.sqrt(tf.cast(2.0, eval(PARAMS["tf_prec"]))) * tf.expand_dims(tf.square(sigma), axis=0) * tf.expand_dims(tf.square(sigma), axis=1)
				* root_inverse_sigma_sum)
	erf_factor = 1 + tf.erf(erf_numerator / erf_denominator)
	overlap_matrix = scaling_factor * exponential_factor * erf_factor / root_inverse_sigma_sum
	return overlap_matrix

def tf_sparse_gauss(dist_tensor, gauss_params):
	exponent = ((tf.square(tf.expand_dims(dist_tensor, axis=-1) - tf.expand_dims(gauss_params[:,0], axis=0)))
				/ (-2.0 * (gauss_params[:,1] ** 2)))
	gaussian_embed = tf.where(tf.greater(exponent, -25.0), tf.exp(exponent), tf.zeros_like(exponent))
	xi = (dist_tensor - 6.0) / (7.0 - 6.0)
	cutoff_factor = 1 - 3 * tf.square(xi) + 2 * tf.pow(xi, 3.0)
	cutoff_factor = tf.where(tf.greater(dist_tensor, 7.0), tf.zeros_like(cutoff_factor), cutoff_factor)
	cutoff_factor = tf.where(tf.less(dist_tensor, 6.0), tf.ones_like(cutoff_factor), cutoff_factor)
	return gaussian_embed * tf.expand_dims(cutoff_factor, axis=-1)

def tf_gauss(dist_tensor, gauss_params, rcut_inner=6.0, rcut_outer=7.0):
	exponent = (tf.square(tf.expand_dims(dist_tensor, axis=-1) - tf.expand_dims(tf.expand_dims(gauss_params[:,0], axis=0), axis=1))) \
				/ (-2.0 * (gauss_params[:,1] ** 2))
	gaussian_embed = tf.where(tf.greater(exponent, -25.0), tf.exp(exponent), tf.zeros_like(exponent))
	gaussian_embed *= tf.expand_dims(tf.where(tf.less(dist_tensor, 1.e-15), tf.zeros_like(dist_tensor),
					tf.ones_like(dist_tensor)), axis=-1)
	xi = (dist_tensor - rcut_inner) / (rcut_outer - rcut_inner)
	cutoff_factor = 1 - 3 * tf.square(xi) + 2 * tf.pow(xi, 3.0)
	cutoff_factor = tf.where(tf.greater(dist_tensor, rcut_outer), tf.zeros_like(cutoff_factor), cutoff_factor)
	cutoff_factor = tf.where(tf.less(dist_tensor, rcut_inner), tf.ones_like(cutoff_factor), cutoff_factor)
	return gaussian_embed * tf.expand_dims(cutoff_factor, axis=-1)

def tf_spherical_harmonics_0(inv_dist_tensor):
	return tf.fill(tf.shape(inv_dist_tensor), tf.constant(0.28209479177387814, dtype=inv_dist_tensor.dtype))

def tf_spherical_harmonics_1(dxyzs, inv_dist_tensor, invariant=False):
	lower_order_harmonics = tf_spherical_harmonics_0(tf.expand_dims(inv_dist_tensor, axis=-1))
	l1_harmonics = 0.4886025119029199 * tf.stack([dxyzs[...,1], dxyzs[...,2], dxyzs[...,0]],
										axis=-1) * tf.expand_dims(inv_dist_tensor, axis=-1)
	if invariant:
		return tf.concat([lower_order_harmonics, tf.norm(l1_harmonics+1.e-16, axis=-1, keepdims=True)], axis=-1)
	else:
		return tf.concat([lower_order_harmonics, l1_harmonics], axis=-1)

def tf_spherical_harmonics_2(dxyzs, inv_dist_tensor, invariant=False):
	if invariant:
		lower_order_harmonics = tf_spherical_harmonics_1(dxyzs, inv_dist_tensor, True)
	else:
		lower_order_harmonics = tf_spherical_harmonics_1(dxyzs, inv_dist_tensor)
	l2_harmonics = tf.stack([(-1.0925484305920792 * dxyzs[...,0] * dxyzs[...,1]),
			(1.0925484305920792 * dxyzs[...,1] * dxyzs[...,2]),
			(-0.31539156525252005 * (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 2. * tf.square(dxyzs[...,2]))),
			(1.0925484305920792 * dxyzs[...,0] * dxyzs[...,2]),
			(0.5462742152960396 * (tf.square(dxyzs[...,0]) - 1. * tf.square(dxyzs[...,1])))], axis=-1) \
			* tf.expand_dims(tf.square(inv_dist_tensor),axis=-1)
	if invariant:
		return tf.concat([lower_order_harmonics, tf.norm(l2_harmonics+1.e-16, axis=-1, keepdims=True)], axis=-1)
	else:
		return tf.concat([lower_order_harmonics, l2_harmonics], axis=-1)

def tf_spherical_harmonics_3(dxyzs, inv_dist_tensor, invariant=False):
	if invariant:
		lower_order_harmonics = tf_spherical_harmonics_2(dxyzs, inv_dist_tensor, True)
	else:
		lower_order_harmonics = tf_spherical_harmonics_2(dxyzs, inv_dist_tensor)
	l3_harmonics = tf.stack([(-0.5900435899266435 * dxyzs[...,1] * (-3. * tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]))),
			(-2.890611442640554 * dxyzs[...,0] * dxyzs[...,1] * dxyzs[...,2]),
			(-0.4570457994644658 * dxyzs[...,1] * (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 4. \
				* tf.square(dxyzs[...,2]))),
			(0.3731763325901154 * dxyzs[...,2] * (-3. * tf.square(dxyzs[...,0]) - 3. * tf.square(dxyzs[...,1]) \
				+ 2. * tf.square(dxyzs[...,2]))),
			(-0.4570457994644658 * dxyzs[...,0] * (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 4. \
				* tf.square(dxyzs[...,2]))),
			(1.445305721320277 * (tf.square(dxyzs[...,0]) - 1. * tf.square(dxyzs[...,1])) * dxyzs[...,2]),
			(0.5900435899266435 * dxyzs[...,0] * (tf.square(dxyzs[...,0]) - 3. * tf.square(dxyzs[...,1])))], axis=-1) \
				* tf.expand_dims(tf.pow(inv_dist_tensor,3),axis=-1)
	if invariant:
		return tf.concat([lower_order_harmonics, tf.norm(l3_harmonics+1.e-16, axis=-1, keepdims=True)], axis=-1)
	else:
		return tf.concat([lower_order_harmonics, l3_harmonics], axis=-1)

def tf_spherical_harmonics_4(dxyzs, inv_dist_tensor, invariant=False):
	if invariant:
		lower_order_harmonics = tf_spherical_harmonics_3(dxyzs, inv_dist_tensor, True)
	else:
		lower_order_harmonics = tf_spherical_harmonics_3(dxyzs, inv_dist_tensor)
	l4_harmonics = tf.stack([(2.5033429417967046 * dxyzs[...,0] * dxyzs[...,1] * (-1. * tf.square(dxyzs[...,0]) \
				+ tf.square(dxyzs[...,1]))),
			(-1.7701307697799304 * dxyzs[...,1] * (-3. * tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1])) * dxyzs[...,2]),
			(0.9461746957575601 * dxyzs[...,0] * dxyzs[...,1] * (tf.square(dxyzs[...,0]) \
				+ tf.square(dxyzs[...,1]) - 6. * tf.square(dxyzs[...,2]))),
			(-0.6690465435572892 * dxyzs[...,1] * dxyzs[...,2] * (3. * tf.square(dxyzs[...,0]) + 3. \
				* tf.square(dxyzs[...,1]) - 4. * tf.square(dxyzs[...,2]))),
			(0.10578554691520431 * (3. * tf.pow(dxyzs[...,0], 4) + 3. * tf.pow(dxyzs[...,1], 4) - 24. \
				* tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 8. * tf.pow(dxyzs[...,2], 4) + 6. \
				* tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 4. * tf.square(dxyzs[...,2])))),
			(-0.6690465435572892 * dxyzs[...,0] * dxyzs[...,2] * (3. * tf.square(dxyzs[...,0]) + 3.
				* tf.square(dxyzs[...,1]) - 4. * tf.square(dxyzs[...,2]))),
			(-0.47308734787878004 * (tf.square(dxyzs[...,0]) - 1. * tf.square(dxyzs[...,1])) \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 6. * tf.square(dxyzs[...,2]))),
			(1.7701307697799304 * dxyzs[...,0] * (tf.square(dxyzs[...,0]) - 3. * tf.square(dxyzs[...,1])) * dxyzs[...,2]),
			(0.6258357354491761 * (tf.pow(dxyzs[...,0], 4) - 6. * tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) \
				+ tf.pow(dxyzs[...,1], 4)))], axis=-1) \
			* tf.expand_dims(tf.pow(inv_dist_tensor,4),axis=-1)
	if invariant:
		return tf.concat([lower_order_harmonics, tf.norm(l4_harmonics+1.e-16, axis=-1, keepdims=True)], axis=-1)
	else:
		return tf.concat([lower_order_harmonics, l4_harmonics], axis=-1)

def tf_spherical_harmonics_5(dxyzs, inv_dist_tensor, invariant=False):
	if invariant:
		lower_order_harmonics = tf_spherical_harmonics_4(dxyzs, inv_dist_tensor, True)
	else:
		lower_order_harmonics = tf_spherical_harmonics_4(dxyzs, inv_dist_tensor)
	l5_harmonics = tf.stack([(0.6563820568401701 * dxyzs[...,1] * (5. * tf.pow(dxyzs[...,0], 4) - 10. \
				* tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) + tf.pow(dxyzs[...,1], 4))),
			(8.302649259524166 * dxyzs[...,0] * dxyzs[...,1] * (-1. * tf.square(dxyzs[...,0]) \
				+ tf.square(dxyzs[...,1])) * dxyzs[...,2]),
			(0.4892382994352504 * dxyzs[...,1] * (-3. * tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1])) \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2]))),
			(4.793536784973324 * dxyzs[...,0] * dxyzs[...,1] * dxyzs[...,2] \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 2. * tf.square(dxyzs[...,2]))),
			(0.45294665119569694 * dxyzs[...,1] * (tf.pow(dxyzs[...,0], 4) + tf.pow(dxyzs[...,1], 4) - 12. \
				* tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 8. * tf.pow(dxyzs[...,2], 4) + 2. \
				* tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 6. * tf.square(dxyzs[...,2])))),
			(0.1169503224534236 * dxyzs[...,2] * (15. * tf.pow(dxyzs[...,0], 4) + 15. * tf.pow(dxyzs[...,1], 4) \
				- 40. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 8. * tf.pow(dxyzs[...,2], 4) + 10. \
				* tf.square(dxyzs[...,0]) * (3. * tf.square(dxyzs[...,1]) - 4. * tf.square(dxyzs[...,2])))),
			(0.45294665119569694 * dxyzs[...,0] * (tf.pow(dxyzs[...,0], 4) + tf.pow(dxyzs[...,1], 4) - 12. \
				* tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 8. * tf.pow(dxyzs[...,2], 4) + 2. \
				* tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 6. * tf.square(dxyzs[...,2])))),
			(-2.396768392486662 * (tf.square(dxyzs[...,0]) - 1. * tf.square(dxyzs[...,1])) * dxyzs[...,2] \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 2. * tf.square(dxyzs[...,2]))),
			(-0.4892382994352504 * dxyzs[...,0] * (tf.square(dxyzs[...,0]) - 3. * tf.square(dxyzs[...,1])) \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2]))),
			(2.0756623148810416 * (tf.pow(dxyzs[...,0], 4) - 6. * tf.square(dxyzs[...,0]) \
				* tf.square(dxyzs[...,1]) + tf.pow(dxyzs[...,1], 4)) * dxyzs[...,2]),
			(0.6563820568401701 * dxyzs[...,0] * (tf.pow(dxyzs[...,0], 4) - 10. \
				* tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) + 5. * tf.pow(dxyzs[...,1], 4)))], axis=-1) \
			* tf.expand_dims(tf.pow(inv_dist_tensor,5),axis=-1)
	if invariant:
		return tf.concat([lower_order_harmonics, tf.norm(l5_harmonics+1.e-16, axis=-1, keepdims=True)], axis=-1)
	else:
		return tf.concat([lower_order_harmonics, l5_harmonics], axis=-1)

def tf_spherical_harmonics_6(dxyzs, inv_dist_tensor, invariant=False):
	if invariant:
		lower_order_harmonics = tf_spherical_harmonics_5(dxyzs, inv_dist_tensor, True)
	else:
		lower_order_harmonics = tf_spherical_harmonics_5(dxyzs, inv_dist_tensor)
	l6_harmonics = tf.stack([(-1.3663682103838286 * dxyzs[...,0] * dxyzs[...,1] * (3. * tf.pow(dxyzs[...,0], 4) \
				- 10. * tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) + 3. * tf.pow(dxyzs[...,1], 4))),
			(2.366619162231752 * dxyzs[...,1] * (5. * tf.pow(dxyzs[...,0], 4) - 10. * tf.square(dxyzs[...,0]) \
				* tf.square(dxyzs[...,1]) + tf.pow(dxyzs[...,1], 4)) * dxyzs[...,2]),
			(2.0182596029148967 * dxyzs[...,0] * dxyzs[...,1] * (tf.square(dxyzs[...,0]) - 1. * tf.square(dxyzs[...,1])) \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 10. * tf.square(dxyzs[...,2]))),
			(0.9212052595149236 * dxyzs[...,1] * (-3. * tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1])) \
				* dxyzs[...,2] * (3. * tf.square(dxyzs[...,0]) + 3. * tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2]))),
			(-0.9212052595149236 * dxyzs[...,0] * dxyzs[...,1] * (tf.pow(dxyzs[...,0], 4) + tf.pow(dxyzs[...,1], 4) \
				- 16. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 16. * tf.pow(dxyzs[...,2], 4) \
				+ 2. * tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2])))),
			(0.5826213625187314 * dxyzs[...,1] * dxyzs[...,2] * (5. * tf.pow(dxyzs[...,0], 4) + 5. * tf.pow(dxyzs[...,1], 4) \
				- 20. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 8. * tf.pow(dxyzs[...,2], 4) \
				+ 10. * tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 2. * tf.square(dxyzs[...,2])))),
			(-0.06356920226762842 * (5. * tf.pow(dxyzs[...,0], 6) + 5. * tf.pow(dxyzs[...,1], 6) - 90. \
				* tf.pow(dxyzs[...,1], 4) * tf.square(dxyzs[...,2]) + 120. * tf.square(dxyzs[...,1]) \
				* tf.pow(dxyzs[...,2], 4) - 16. * tf.pow(dxyzs[...,2], 6) + 15. * tf.pow(dxyzs[...,0], 4) \
				* (tf.square(dxyzs[...,1]) - 6. * tf.square(dxyzs[...,2])) + 15. * tf.square(dxyzs[...,0]) \
				* (tf.pow(dxyzs[...,1], 4) - 12. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 8. * tf.pow(dxyzs[...,2], 4)))),
			(0.5826213625187314 * dxyzs[...,0] * dxyzs[...,2] * (5. * tf.pow(dxyzs[...,0], 4) + 5. \
				* tf.pow(dxyzs[...,1], 4) - 20. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 8. \
				* tf.pow(dxyzs[...,2], 4) + 10. * tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 2. \
				* tf.square(dxyzs[...,2])))),
			(0.4606026297574618 * (tf.square(dxyzs[...,0]) - 1. * tf.square(dxyzs[...,1])) * (tf.pow(dxyzs[...,0], 4) \
				+ tf.pow(dxyzs[...,1], 4) - 16. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 16. \
				* tf.pow(dxyzs[...,2], 4) + 2. * tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 8. \
				* tf.square(dxyzs[...,2])))),
			(-0.9212052595149236 * dxyzs[...,0] * (tf.square(dxyzs[...,0]) - 3. * tf.square(dxyzs[...,1])) * dxyzs[...,2] \
				* (3. * tf.square(dxyzs[...,0]) + 3. * tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2]))),
			(-0.5045649007287242 * (tf.pow(dxyzs[...,0], 4) - 6. * tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) \
				+ tf.pow(dxyzs[...,1], 4)) * (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 10. * tf.square(dxyzs[...,2]))),
			(2.366619162231752 * dxyzs[...,0] * (tf.pow(dxyzs[...,0], 4) - 10. * tf.square(dxyzs[...,0]) \
				* tf.square(dxyzs[...,1]) + 5. * tf.pow(dxyzs[...,1], 4)) * dxyzs[...,2]),
			(0.6831841051919143 * (tf.pow(dxyzs[...,0], 6) - 15. * tf.pow(dxyzs[...,0], 4) * tf.square(dxyzs[...,1]) \
				+ 15. * tf.square(dxyzs[...,0]) * tf.pow(dxyzs[...,1], 4) - 1. * tf.pow(dxyzs[...,1], 6)))], axis=-1) \
			* tf.expand_dims(tf.pow(inv_dist_tensor,6),axis=-1)
	if invariant:
		return tf.concat([lower_order_harmonics, tf.norm(l6_harmonics+1.e-16, axis=-1, keepdims=True)], axis=-1)
	else:
		return tf.concat([lower_order_harmonics, l6_harmonics], axis=-1)

def tf_spherical_harmonics_7(dxyzs, inv_dist_tensor, invariant=False):
	if invariant:
		lower_order_harmonics = tf_spherical_harmonics_6(dxyzs, inv_dist_tensor, True)
	else:
		lower_order_harmonics = tf_spherical_harmonics_6(dxyzs, inv_dist_tensor)
	l7_harmonics = tf.stack([(-0.7071627325245962 * dxyzs[...,1] * (-7. * tf.pow(dxyzs[...,0], 6) + 35. \
				* tf.pow(dxyzs[...,0], 4) * tf.square(dxyzs[...,1]) - 21. * tf.square(dxyzs[...,0]) \
				* tf.pow(dxyzs[...,1], 4) + tf.pow(dxyzs[...,1], 6))),
			(-5.291921323603801 * dxyzs[...,0] * dxyzs[...,1] * (3. * tf.pow(dxyzs[...,0], 4) - 10. \
				* tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) + 3. * tf.pow(dxyzs[...,1], 4)) * dxyzs[...,2]),
			(-0.5189155787202604 * dxyzs[...,1] * (5. * tf.pow(dxyzs[...,0], 4) - 10. \
				* tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) + tf.pow(dxyzs[...,1], 4)) \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 12. * tf.square(dxyzs[...,2]))),
			(4.151324629762083 * dxyzs[...,0] * dxyzs[...,1] * (tf.square(dxyzs[...,0]) - 1. \
				* tf.square(dxyzs[...,1])) * dxyzs[...,2] * (3. * tf.square(dxyzs[...,0]) + 3. \
				* tf.square(dxyzs[...,1]) - 10. * tf.square(dxyzs[...,2]))),
			(-0.15645893386229404 * dxyzs[...,1] * (-3. * tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1])) \
				* (3. * tf.pow(dxyzs[...,0], 4) + 3. * tf.pow(dxyzs[...,1], 4) - 60. * tf.square(dxyzs[...,1]) \
				* tf.square(dxyzs[...,2]) + 80. * tf.pow(dxyzs[...,2], 4) + 6. * tf.square(dxyzs[...,0]) \
				* (tf.square(dxyzs[...,1]) - 10. * tf.square(dxyzs[...,2])))),
			(-0.4425326924449826 * dxyzs[...,0] * dxyzs[...,1] * dxyzs[...,2] * (15. * tf.pow(dxyzs[...,0], 4) \
				+ 15. * tf.pow(dxyzs[...,1], 4) - 80. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 48. \
				* tf.pow(dxyzs[...,2], 4) + 10. * tf.square(dxyzs[...,0]) * (3. * tf.square(dxyzs[...,1]) - 8. \
				* tf.square(dxyzs[...,2])))),
			(-0.0903316075825173 * dxyzs[...,1] * (5. * tf.pow(dxyzs[...,0], 6) + 5. * tf.pow(dxyzs[...,1], 6) - 120. \
				* tf.pow(dxyzs[...,1], 4) * tf.square(dxyzs[...,2]) + 240. * tf.square(dxyzs[...,1]) \
				* tf.pow(dxyzs[...,2], 4) - 64. * tf.pow(dxyzs[...,2], 6) + 15. * tf.pow(dxyzs[...,0], 4) \
				* (tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2])) + 15. * tf.square(dxyzs[...,0]) \
				* (tf.pow(dxyzs[...,1], 4) - 16. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 16. \
				* tf.pow(dxyzs[...,2], 4)))),
			(0.06828427691200495 * dxyzs[...,2] * (-35. * tf.pow(dxyzs[...,0], 6) - 35. * tf.pow(dxyzs[...,1], 6) \
				+ 210. * tf.pow(dxyzs[...,1], 4) * tf.square(dxyzs[...,2]) - 168. * tf.square(dxyzs[...,1]) \
				* tf.pow(dxyzs[...,2], 4) + 16. * tf.pow(dxyzs[...,2], 6) - 105. * tf.pow(dxyzs[...,0], 4) \
				* (tf.square(dxyzs[...,1]) - 2. * tf.square(dxyzs[...,2])) - 21. * tf.square(dxyzs[...,0]) \
				* (5. * tf.pow(dxyzs[...,1], 4) - 20. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) \
				+ 8. * tf.pow(dxyzs[...,2], 4)))),
			(-0.0903316075825173 * dxyzs[...,0] * (5. * tf.pow(dxyzs[...,0], 6) + 5. * tf.pow(dxyzs[...,1], 6) \
				- 120. * tf.pow(dxyzs[...,1], 4) * tf.square(dxyzs[...,2]) + 240. * tf.square(dxyzs[...,1]) \
				* tf.pow(dxyzs[...,2], 4) - 64. * tf.pow(dxyzs[...,2], 6) + 15. * tf.pow(dxyzs[...,0], 4) \
				* (tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2])) + 15. * tf.square(dxyzs[...,0]) \
				* (tf.pow(dxyzs[...,1], 4) - 16. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 16. \
				* tf.pow(dxyzs[...,2], 4)))),
			(0.2212663462224913 * (tf.square(dxyzs[...,0]) - 1. * tf.square(dxyzs[...,1])) * dxyzs[...,2] \
				* (15. * tf.pow(dxyzs[...,0], 4) + 15. * tf.pow(dxyzs[...,1], 4) - 80. * tf.square(dxyzs[...,1]) \
				* tf.square(dxyzs[...,2]) + 48. * tf.pow(dxyzs[...,2], 4) + 10. * tf.square(dxyzs[...,0]) \
				* (3. * tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2])))),
			(0.15645893386229404 * dxyzs[...,0] * (tf.square(dxyzs[...,0]) - 3. * tf.square(dxyzs[...,1])) \
				* (3. * tf.pow(dxyzs[...,0], 4) + 3. * tf.pow(dxyzs[...,1], 4) - 60. * tf.square(dxyzs[...,1]) \
				* tf.square(dxyzs[...,2]) + 80. * tf.pow(dxyzs[...,2], 4) + 6. * tf.square(dxyzs[...,0]) \
				* (tf.square(dxyzs[...,1]) - 10. * tf.square(dxyzs[...,2])))),
			(-1.0378311574405208 * (tf.pow(dxyzs[...,0], 4) - 6. * tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) \
				+ tf.pow(dxyzs[...,1], 4)) * dxyzs[...,2] * (3. * tf.square(dxyzs[...,0]) \
				+ 3. * tf.square(dxyzs[...,1]) - 10. * tf.square(dxyzs[...,2]))),
			(-0.5189155787202604 * dxyzs[...,0] * (tf.pow(dxyzs[...,0], 4) - 10. * tf.square(dxyzs[...,0]) \
				* tf.square(dxyzs[...,1]) + 5. * tf.pow(dxyzs[...,1], 4)) * (tf.square(dxyzs[...,0]) \
				+ tf.square(dxyzs[...,1]) - 12. * tf.square(dxyzs[...,2]))),
			(2.6459606618019005 * (tf.pow(dxyzs[...,0], 6) - 15. * tf.pow(dxyzs[...,0], 4) * tf.square(dxyzs[...,1]) \
				+ 15. * tf.square(dxyzs[...,0]) * tf.pow(dxyzs[...,1], 4) - 1. * tf.pow(dxyzs[...,1], 6)) * dxyzs[...,2]),
			(0.7071627325245962 * dxyzs[...,0] * (tf.pow(dxyzs[...,0], 6) - 21. * tf.pow(dxyzs[...,0], 4) \
				* tf.square(dxyzs[...,1]) + 35. * tf.square(dxyzs[...,0]) * tf.pow(dxyzs[...,1], 4) - 7. \
				* tf.pow(dxyzs[...,1], 6)))], axis=-1) \
			* tf.expand_dims(tf.pow(inv_dist_tensor,7),axis=-1)
	if invariant:
		return tf.concat([lower_order_harmonics, tf.norm(l7_harmonics+1.e-16, axis=-1, keepdims=True)], axis=-1)
	else:
		return tf.concat([lower_order_harmonics, l7_harmonics], axis=-1)

def tf_spherical_harmonics_8(dxyzs, inv_dist_tensor, invariant=False):
	if invariant:
		lower_order_harmonics = tf_spherical_harmonics_7(dxyzs, inv_dist_tensor, True)
	else:
		lower_order_harmonics = tf_spherical_harmonics_7(dxyzs, inv_dist_tensor)
	l8_harmonics = tf.stack([(-5.831413281398639 * dxyzs[...,0] * dxyzs[...,1] * (tf.pow(dxyzs[...,0], 6) \
				- 7. * tf.pow(dxyzs[...,0], 4) * tf.square(dxyzs[...,1]) + 7. * tf.square(dxyzs[...,0]) \
				* tf.pow(dxyzs[...,1], 4) - 1. * tf.pow(dxyzs[...,1], 6))),
			(-2.9157066406993195 * dxyzs[...,1] * (-7. * tf.pow(dxyzs[...,0], 6) + 35. * tf.pow(dxyzs[...,0], 4) \
				* tf.square(dxyzs[...,1]) - 21. * tf.square(dxyzs[...,0]) * tf.pow(dxyzs[...,1], 4) \
				+ tf.pow(dxyzs[...,1], 6)) * dxyzs[...,2]),
			(1.0646655321190852 * dxyzs[...,0] * dxyzs[...,1] * (3. * tf.pow(dxyzs[...,0], 4) - 10. \
				* tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) + 3. * tf.pow(dxyzs[...,1], 4)) \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 14. * tf.square(dxyzs[...,2]))),
			(-3.449910622098108 * dxyzs[...,1] * (5. * tf.pow(dxyzs[...,0], 4) - 10. * tf.square(dxyzs[...,0]) \
				* tf.square(dxyzs[...,1]) + tf.pow(dxyzs[...,1], 4)) * dxyzs[...,2] * (tf.square(dxyzs[...,0]) \
				+ tf.square(dxyzs[...,1]) - 4. * tf.square(dxyzs[...,2]))),
			(-1.9136660990373227 * dxyzs[...,0] * dxyzs[...,1] * (tf.square(dxyzs[...,0]) - 1. \
				* tf.square(dxyzs[...,1])) * (tf.pow(dxyzs[...,0], 4) + tf.pow(dxyzs[...,1], 4) - 24. \
				* tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 40. * tf.pow(dxyzs[...,2], 4) + 2. \
				* tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 12. * tf.square(dxyzs[...,2])))),
			(-1.2352661552955442 * dxyzs[...,1] * (-3. * tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1])) \
				* dxyzs[...,2] * (3. * tf.pow(dxyzs[...,0], 4) + 3. * tf.pow(dxyzs[...,1], 4) - 20. \
				* tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 16. * tf.pow(dxyzs[...,2], 4) \
				+ tf.square(dxyzs[...,0]) * (6. * tf.square(dxyzs[...,1]) - 20. * tf.square(dxyzs[...,2])))),
			(0.912304516869819 * dxyzs[...,0] * dxyzs[...,1] * (tf.pow(dxyzs[...,0], 6) + tf.pow(dxyzs[...,1], 6) \
				- 30. * tf.pow(dxyzs[...,1], 4) * tf.square(dxyzs[...,2]) + 80. * tf.square(dxyzs[...,1]) \
				* tf.pow(dxyzs[...,2], 4) - 32. * tf.pow(dxyzs[...,2], 6) + 3. * tf.pow(dxyzs[...,0], 4) \
				* (tf.square(dxyzs[...,1]) - 10. * tf.square(dxyzs[...,2])) + tf.square(dxyzs[...,0]) \
				* (3. * tf.pow(dxyzs[...,1], 4) - 60. * tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 80. \
				* tf.pow(dxyzs[...,2], 4)))),
			(-0.10904124589877995 * dxyzs[...,1] * dxyzs[...,2] * (35. * tf.pow(dxyzs[...,0], 6) + 35. \
				* tf.pow(dxyzs[...,1], 6) - 280. * tf.pow(dxyzs[...,1], 4) * tf.square(dxyzs[...,2]) + 336. \
				* tf.square(dxyzs[...,1]) * tf.pow(dxyzs[...,2], 4) - 64. * tf.pow(dxyzs[...,2], 6) + 35. \
				* tf.pow(dxyzs[...,0], 4) * (3. * tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2])) + 7. \
				* tf.square(dxyzs[...,0]) * (15. * tf.pow(dxyzs[...,1], 4) - 80. * tf.square(dxyzs[...,1]) \
				* tf.square(dxyzs[...,2]) + 48. * tf.pow(dxyzs[...,2], 4)))),
			(0.009086770491564996 * (35. * tf.pow(dxyzs[...,0], 8) + 35. * tf.pow(dxyzs[...,1], 8) - 1120. \
				* tf.pow(dxyzs[...,1], 6) * tf.square(dxyzs[...,2]) + 3360. * tf.pow(dxyzs[...,1], 4) \
				* tf.pow(dxyzs[...,2], 4) - 1792. * tf.square(dxyzs[...,1]) * tf.pow(dxyzs[...,2], 6) + 128. \
				* tf.pow(dxyzs[...,2], 8) + 140. * tf.pow(dxyzs[...,0], 6) * (tf.square(dxyzs[...,1]) - 8. \
				* tf.square(dxyzs[...,2])) + 210. * tf.pow(dxyzs[...,0], 4) * (tf.pow(dxyzs[...,1], 4) - 16. \
				* tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 16. * tf.pow(dxyzs[...,2], 4)) + 28. \
				* tf.square(dxyzs[...,0]) * (5. * tf.pow(dxyzs[...,1], 6) - 120. * tf.pow(dxyzs[...,1], 4) \
				* tf.square(dxyzs[...,2]) + 240. * tf.square(dxyzs[...,1]) * tf.pow(dxyzs[...,2], 4) - 64. \
				* tf.pow(dxyzs[...,2], 6)))),
			(-0.10904124589877995 * dxyzs[...,0] * dxyzs[...,2] * (35. * tf.pow(dxyzs[...,0], 6) + 35. \
				* tf.pow(dxyzs[...,1], 6) - 280. * tf.pow(dxyzs[...,1], 4) * tf.square(dxyzs[...,2]) + 336. \
				* tf.square(dxyzs[...,1]) * tf.pow(dxyzs[...,2], 4) - 64. * tf.pow(dxyzs[...,2], 6) + 35. \
				* tf.pow(dxyzs[...,0], 4) * (3. * tf.square(dxyzs[...,1]) - 8. * tf.square(dxyzs[...,2])) + 7. \
				* tf.square(dxyzs[...,0]) * (15. * tf.pow(dxyzs[...,1], 4) - 80. * tf.square(dxyzs[...,1]) \
				* tf.square(dxyzs[...,2]) + 48. * tf.pow(dxyzs[...,2], 4)))),
			(-0.4561522584349095 * (tf.square(dxyzs[...,0]) - 1. * tf.square(dxyzs[...,1])) * (tf.pow(dxyzs[...,0], 6) \
				+ tf.pow(dxyzs[...,1], 6) - 30. * tf.pow(dxyzs[...,1], 4) * tf.square(dxyzs[...,2]) + 80. \
				* tf.square(dxyzs[...,1]) * tf.pow(dxyzs[...,2], 4) - 32. * tf.pow(dxyzs[...,2], 6) + 3. \
				* tf.pow(dxyzs[...,0], 4) * (tf.square(dxyzs[...,1]) - 10. * tf.square(dxyzs[...,2])) \
				+ tf.square(dxyzs[...,0]) * (3. * tf.pow(dxyzs[...,1], 4) - 60. * tf.square(dxyzs[...,1]) \
				* tf.square(dxyzs[...,2]) + 80. * tf.pow(dxyzs[...,2], 4)))),
			(1.2352661552955442 * dxyzs[...,0] * (tf.square(dxyzs[...,0]) - 3. * tf.square(dxyzs[...,1])) \
				* dxyzs[...,2] * (3. * tf.pow(dxyzs[...,0], 4) + 3. * tf.pow(dxyzs[...,1], 4) - 20. \
				* tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 16. * tf.pow(dxyzs[...,2], 4) \
				+ tf.square(dxyzs[...,0]) * (6. * tf.square(dxyzs[...,1]) - 20. * tf.square(dxyzs[...,2])))),
			(0.47841652475933066 * (tf.pow(dxyzs[...,0], 4) - 6. * tf.square(dxyzs[...,0]) * tf.square(dxyzs[...,1]) \
				+ tf.pow(dxyzs[...,1], 4)) * (tf.pow(dxyzs[...,0], 4) + tf.pow(dxyzs[...,1], 4) - 24. \
				* tf.square(dxyzs[...,1]) * tf.square(dxyzs[...,2]) + 40. * tf.pow(dxyzs[...,2], 4) + 2. \
				* tf.square(dxyzs[...,0]) * (tf.square(dxyzs[...,1]) - 12. * tf.square(dxyzs[...,2])))),
			(-3.449910622098108 * dxyzs[...,0] * (tf.pow(dxyzs[...,0], 4) - 10. * tf.square(dxyzs[...,0]) \
				* tf.square(dxyzs[...,1]) + 5. * tf.pow(dxyzs[...,1], 4)) * dxyzs[...,2] * (tf.square(dxyzs[...,0]) \
				+ tf.square(dxyzs[...,1]) - 4. * tf.square(dxyzs[...,2]))),
			(-0.5323327660595426 * (tf.pow(dxyzs[...,0], 6) - 15. * tf.pow(dxyzs[...,0], 4) * tf.square(dxyzs[...,1]) \
				+ 15. * tf.square(dxyzs[...,0]) * tf.pow(dxyzs[...,1], 4) - 1. * tf.pow(dxyzs[...,1], 6)) \
				* (tf.square(dxyzs[...,0]) + tf.square(dxyzs[...,1]) - 14. * tf.square(dxyzs[...,2]))),
			(2.9157066406993195 * dxyzs[...,0] * (tf.pow(dxyzs[...,0], 6) - 21. * tf.pow(dxyzs[...,0], 4) \
				* tf.square(dxyzs[...,1]) + 35. * tf.square(dxyzs[...,0]) * tf.pow(dxyzs[...,1], 4) - 7. \
				* tf.pow(dxyzs[...,1], 6)) * dxyzs[...,2]),
			(0.7289266601748299 * (tf.pow(dxyzs[...,0], 8) - 28. * tf.pow(dxyzs[...,0], 6) * tf.square(dxyzs[...,1]) \
				+ 70. * tf.pow(dxyzs[...,0], 4) * tf.pow(dxyzs[...,1], 4) - 28. * tf.square(dxyzs[...,0]) \
				* tf.pow(dxyzs[...,1], 6) + tf.pow(dxyzs[...,1], 8)))], axis=-1) \
			* tf.expand_dims(tf.pow(inv_dist_tensor,8),axis=-1)
	if invariant:
		return tf.concat([lower_order_harmonics, tf.norm(l8_harmonics+1.e-16, axis=-1, keepdims=True)], axis=-1)
	else:
		return tf.concat([lower_order_harmonics, l8_harmonics], axis=-1)

def tf_spherical_harmonics(dxyzs, dist_tensor, max_l, invariant=False):
	"""
	Args:
		dxyzs: (...) X MaxNAtom X MaxNAtom X 3 (differenced from center of embedding
				ie: ... X i X i = (0.,0.,0.))
		dist_tensor: just tf.norm of the above.
		max_l : integer, maximum angular momentum.
		invariant: whether to return just total angular momentum of a given l.
	Returns:
		(...) X MaxNAtom X MaxNAtom X {NSH = (max_l+1)^2}
	"""
	inv_dist_tensor = tf.where(tf.greater(dist_tensor, 1.e-9), tf.reciprocal(dist_tensor), tf.zeros_like(dist_tensor))
	if max_l == 8:
		harmonics = tf_spherical_harmonics_8(dxyzs, inv_dist_tensor, invariant)
	elif max_l == 7:
		harmonics = tf_spherical_harmonics_7(dxyzs, inv_dist_tensor, invariant)
	elif max_l == 6:
		harmonics = tf_spherical_harmonics_6(dxyzs, inv_dist_tensor, invariant)
	elif max_l == 5:
		harmonics = tf_spherical_harmonics_5(dxyzs, inv_dist_tensor, invariant)
	elif max_l == 4:
		harmonics = tf_spherical_harmonics_4(dxyzs, inv_dist_tensor, invariant)
	elif max_l == 3:
		harmonics = tf_spherical_harmonics_3(dxyzs, inv_dist_tensor, invariant)
	elif max_l == 2:
		harmonics = tf_spherical_harmonics_2(dxyzs, inv_dist_tensor, invariant)
	elif max_l == 1:
		harmonics = tf_spherical_harmonics_1(dxyzs, inv_dist_tensor, invariant)
	elif max_l == 0:
		harmonics = tf_spherical_harmonics_0(inv_dist_tensor)
	else:
		raise Exception("Spherical Harmonics only implemented up to l=8. Choose a lower order")
	return harmonics

#
# JOHN: we should write the following routines.
# this will just require cooking up the overlap matrix.
# which is a simple integral.
#

def gaussian_spherical_harmonic_fwd(xyzs_,lmax=5):
	"""
	embed delta functions at xyz into GauSH representation.
	"""
	return

def gaussian_spherical_harmonic_S(gauss_params,lmax=5):
	"""
	Makes an overlap matrix of the gaussian spherical functions
	"""
	return

def gaussian_spherical_harmonic_rev(vec_,samps_,lmax=5):
	"""
	Reverses the embedding of a gauSH vector onto a sampling grid.
	"""
	return

def tf_gaush_element_channel(xyzs, Zs, elements, gauss_params, l_max):
	"""
	Encodes atoms into a gaussians * spherical harmonics embedding
	cast into element channels. Works on a batch of molecules.

	Args:
		xyzs (tf.float): NMol x MaxNAtom x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtom atomic number tensor
		element (int): element to return embedding/labels for
		gauss_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use

	Returns:
		embedding (tf.float): atom embeddings for element
		molecule_indices (tf.float): mapping between atoms and molecules.
	"""
	num_elements = elements.get_shape().as_list()[0]
	num_mols = Zs.get_shape().as_list()[0]
	padding_mask = tf.where(tf.not_equal(Zs, 0))

	dxyzs = tf.gather_nd(xyzs, padding_mask)
	dist_tensor = tf.norm(dxyzs+1.e-16,axis=-1)
	gauss = tf_gauss(dist_tensor, gauss_params)
	harmonics = tf_spherical_harmonics(dxyzs, dist_tensor, l_max)
	channel_scatter = tf.gather(tf.equal(tf.expand_dims(Zs, axis=-1), elements), padding_mask[:,0])
	channel_scatter = tf.where(channel_scatter, tf.ones_like(channel_scatter, dtype=eval(PARAMS["tf_prec"])),
					tf.zeros_like(channel_scatter, dtype=eval(PARAMS["tf_prec"])))
	channel_gauss = tf.expand_dims(gauss, axis=-2) * tf.expand_dims(channel_scatter, axis=-1)
	channel_harmonics = tf.expand_dims(harmonics, axis=-2) * tf.expand_dims(channel_scatter, axis=-1)
	embeds = tf.reshape(tf.einsum('ijkg,ijkl->ikgl', channel_gauss, channel_harmonics),
			[tf.shape(padding_mask)[0], -1])
	partition_idx = tf.cast(tf.where(tf.equal(tf.expand_dims(tf.gather_nd(Zs, padding_mask), axis=-1),
						tf.expand_dims(elements, axis=0)))[:,1], tf.int32)
	with tf.device('/cpu:0'):
		embeds = tf.dynamic_partition(embeds, partition_idx, num_elements)
		mol_idx = tf.dynamic_partition(padding_mask, partition_idx, num_elements)
	return embeds, mol_idx

def tf_sparse_gaush_element_channel(xyzs, Zs, pair_Zs, elements, gauss_params, l_max):
	"""
	Sparse version of tf_gauss_harmonics_echannel.
	Encodes atoms into a gaussians * spherical harmonics embedding
	cast into element channels. Works on a batch of molecules.

	Args:
		xyzs (tf.float): NMol x MaxNAtom x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtom atomic number tensor
		pairs (tf.int32): NMol x MaxNAtom x MaxNNeighbors neighbor index tensor
		element (int): element to return embedding/labels for
		gauss_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use

	Returns:
		embedding (tf.float): atom embeddings for element
		molecule_indices (tf.float): mapping between atoms and molecules.
	"""
	num_elements = elements.get_shape().as_list()[0]
	num_mols = Zs.get_shape().as_list()[0]
	padding_mask = tf.where(tf.not_equal(Zs, 0))

	dist_tensor = tf.norm(xyzs+1.e-16,axis=-1)
	gauss = tf_gauss(dist_tensor, gauss_params)
	harmonics = tf_spherical_harmonics(xyzs, dist_tensor, l_max)
	channel_scatter = tf.equal(tf.expand_dims(pair_Zs, axis=-1), elements)
	channel_scatter = tf.where(channel_scatter, tf.ones_like(channel_scatter, dtype=eval(PARAMS["tf_prec"])),
					tf.zeros_like(channel_scatter, dtype=eval(PARAMS["tf_prec"])))
	channel_gauss = tf.expand_dims(gauss, axis=-2) * tf.expand_dims(channel_scatter, axis=-1)
	channel_harmonics = tf.expand_dims(harmonics, axis=-2) * tf.expand_dims(channel_scatter, axis=-1)
	embeds = tf.reshape(tf.einsum('ijkg,ijkl->ikgl', channel_gauss, channel_harmonics),
			[tf.shape(padding_mask)[0], -1])
	partition_idx = tf.cast(tf.where(tf.equal(tf.expand_dims(tf.gather_nd(Zs, padding_mask), axis=-1),
						tf.expand_dims(elements, axis=0)))[:,1], tf.int32)
	with tf.device('/cpu:0'):
		embeds = tf.dynamic_partition(embeds, partition_idx, num_elements)
		mol_idx = tf.dynamic_partition(padding_mask, partition_idx, num_elements)
	return embeds, mol_idx

def tf_gaush_embed_channel(xyzs, Zs, elements, gauss_params, l_max, embed_factor):
	"""
	Encodes atoms into a gaussians * spherical harmonics embedding
	cast into channels with embedding factors. Works on a batch of molecules.

	Args:
		xyzs (tf.float): NMol x MaxNAtom x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtom atomic number tensor
		elements (tf.int32): NElements tensor of unique atomic numbers
		gauss_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use
		embed_factor (tf.float): NChannels x NElements tensor of embedding factors

	Returns:
		embeds (tf.float): list of atomic embedding tensors for each element
		mol_idx (tf.int32): list of molecule and atom indices matching embeds
	"""
	num_elements = elements.get_shape().as_list()[0]
	num_mols = Zs.get_shape().as_list()[0]
	num_channels = embed_factor.get_shape().as_list()[0]
	padding_mask = tf.where(tf.not_equal(Zs, 0))
	dist_tensor = tf.norm(xyzs+1.e-16,axis=-1)
	gauss = tf_gauss(dist_tensor, gauss_params)
	harmonics = tf_spherical_harmonics(xyzs, dist_tensor, l_max)
	embeds = tf.reshape(tf.einsum('ijk,ijl->ijkl', gauss, harmonics), [tf.shape(xyzs)[0], tf.shape(xyzs)[1], -1])
	pair_Zs = tf.gather(Zs, padding_mask[:,0])
	channel_factors = tf.reshape(tf.gather(embed_factor, tf.reshape(pair_Zs, [-1])), [tf.shape(xyzs)[0], tf.shape(xyzs)[1], 4])
	embeds = tf.reduce_sum(tf.expand_dims(embeds, axis=-1) * tf.expand_dims(channel_factors, axis=-2), axis=1)
	return embeds

def tf_gaush_element_channelv3(xyzs, Zs, elements, gauss_params, l_max):
	"""
	Encodes atoms into a gaussians * spherical harmonics embedding
	cast into element channels. Works on a batch of molecules.

	Args:
		xyzs (tf.float): NMol x MaxNAtom x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtom atomic number tensor
		element (int): element to return embedding/labels for
		gauss_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use

	Returns:
		embedding (tf.float): atom embeddings for element
		molecule_indices (tf.float): mapping between atoms and molecules.
	"""
	num_elements = elements.get_shape().as_list()[0]
	num_mols = Zs.get_shape().as_list()[0]
	padding_mask = tf.where(tf.not_equal(Zs, 0))

	dist_tensor = tf.norm(xyzs+1.e-16,axis=-1)
	gauss = tf_gauss(dist_tensor, gauss_params)
	harmonics = tf_spherical_harmonics(xyzs, dist_tensor, l_max)
	channel_scatter = tf.gather(tf.equal(tf.expand_dims(Zs, axis=-1), elements), padding_mask[:,0])
	channel_scatter = tf.where(channel_scatter, tf.ones_like(channel_scatter, dtype=eval(PARAMS["tf_prec"])),
					tf.zeros_like(channel_scatter, dtype=eval(PARAMS["tf_prec"])))
	channel_gauss = tf.expand_dims(gauss, axis=-2) * tf.expand_dims(channel_scatter, axis=-1)
	channel_harmonics = tf.expand_dims(harmonics, axis=-2) * tf.expand_dims(channel_scatter, axis=-1)
	embeds = tf.reshape(tf.einsum('ijkg,ijkl->ikgl', channel_gauss, channel_harmonics),
			[tf.shape(padding_mask)[0], -1])
	partition_idx = tf.cast(tf.where(tf.equal(tf.expand_dims(tf.gather_nd(Zs, padding_mask), axis=-1),
						tf.expand_dims(elements, axis=0)))[:,1], tf.int32)
	with tf.device('/cpu:0'):
		embeds = tf.dynamic_partition(embeds, partition_idx, num_elements)
		mol_idx = tf.dynamic_partition(padding_mask, partition_idx, num_elements)
	return embeds, mol_idx

def tf_random_rotate(xyzs, rot_params, labels = None, return_matrix = False):
	"""
	Rotates molecules and optionally labels in a uniformly random fashion

	Args:
		xyzs (tf.float): NMol x MaxNAtom x 3 coordinates tensor
		labels (tf.float, optional): NMol x MaxNAtom x label shape tensor of learning targets
		return_matrix (bool): Returns rotation tensor if True

	Returns:
		new_xyzs (tf.float): NMol x MaxNAtom x 3 coordinates tensor of randomly rotated molecules
		new_labels (tf.float): NMol x MaxNAtom x label shape tensor of randomly rotated learning targets
	"""
	r = tf.sqrt(rot_params[...,2])
	v = tf.stack([tf.sin(rot_params[...,1]) * r, tf.cos(rot_params[...,1]) * r, tf.sqrt(2.0 - rot_params[...,2])], axis=-1)
	zero_tensor = tf.zeros_like(rot_params[...,1])

	R1 = tf.stack([tf.cos(rot_params[...,0]), tf.sin(rot_params[...,0]), zero_tensor], axis=-1)
	R2 = tf.stack([-tf.sin(rot_params[...,0]), tf.cos(rot_params[...,0]), zero_tensor], axis=-1)
	R3 = tf.stack([zero_tensor, zero_tensor, tf.ones_like(rot_params[...,1])], axis=-1)
	R = tf.stack([R1, R2, R3], axis=-2)
	M = tf.matmul((tf.expand_dims(v, axis=-2) * tf.expand_dims(v, axis=-1)) - tf.eye(3, dtype=eval(PARAMS["tf_prec"])), R)
	# return xyzs
	new_xyzs = tf.einsum("lij,lkj->lki", M, xyzs)
	if labels != None:
		new_labels = tf.einsum("lij,lkj->lki",M, (xyzs + labels)) - new_xyzs
		if return_matrix:
			return new_xyzs, new_labels, M
		else:
			return new_xyzs, new_labels
	elif return_matrix:
		return new_xyzs, M
	else:
		return new_xyzs

def tf_rotate(xyzs, axis, angle):
	"""
	Rotates molecules and optionally labels in a uniformly random fashion

	Args:
		xyzs (tf.float): NMol x MaxNAtom x 3 coordinates tensor
		labels (tf.float, optional): NMol x MaxNAtom x label shape tensor of learning targets
		return_matrix (bool): Returns rotation tensor if True

	Returns:
		new_xyzs (tf.float): NMol x MaxNAtom x 3 coordinates tensor of randomly rotated molecules
		new_labels (tf.float): NMol x MaxNAtom x label shape tensor of randomly rotated learning targets
	"""
	axis = tf.tile(tf.expand_dims(axis / tf.norm(axis, axis=-1, keepdims=True), axis=-2), [1, tf.shape(xyzs)[1], 1])
	angle = tf.reshape(angle, [tf.shape(angle)[0], 1, 1])
	term1 = xyzs * tf.cos(angle)
	term2 = tf.cross(axis, xyzs) * tf.sin(angle)
	term3 = axis * (1 - tf.cos(angle)) * tf.reduce_sum(axis * xyzs, axis=-1, keepdims=True)
	new_xyzs = term1 + term2 + term3
	return new_xyzs

def tf_coulomb_dsf_elu(xyzs, charges, Radpair, elu_width, dsf_alpha, cutoff_dist):
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
	xyzs *= BOHRPERA
	elu_width *= BOHRPERA
	dsf_alpha /= BOHRPERA
	cutoff_dist *= BOHRPERA
	inp_shp = tf.shape(xyzs)
	num_mol = tf.cast(tf.shape(xyzs)[0], dtype=tf.int64)
	num_pairs = tf.cast(tf.shape(Radpair)[0], tf.int64)
	elu_shift, elu_alpha = tf_dsf_potential(elu_width, cutoff_dist, dsf_alpha, return_grad=True)

	Rij = DifferenceVectorsLinear(xyzs + 1e-16, Radpair)
	RijRij2 = tf.norm(Rij, axis=1)
	qij = tf.gather_nd(charges, Radpair[:,:2]) * tf.gather_nd(charges, Radpair[:,::2])

	coulomb_potential = qij * tf.where(tf.greater(RijRij2, elu_width), tf_dsf_potential(RijRij2, cutoff_dist, dsf_alpha),
						elu_alpha * (tf.exp(RijRij2 - elu_width) - 1.0) + elu_shift)

	range_index = tf.range(num_pairs, dtype=tf.int64)
	mol_index = tf.cast(Radpair[:,0], dtype=tf.int64)
	sparse_index = tf.cast(tf.stack([mol_index, range_index], axis=1), tf.int64)
	sp_atomoutputs = tf.SparseTensor(sparse_index, coulomb_potential, [num_mol, num_pairs])
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def tf_dsf_potential(dists, cutoff_dist, dsf_alpha, return_grad=False):
	dsf_potential = tf.erfc(dsf_alpha * dists) / dists - tf.erfc(dsf_alpha * cutoff_dist) / cutoff_dist \
					+ (dists - cutoff_dist) * (tf.erfc(dsf_alpha * cutoff_dist) / tf.square(cutoff_dist) \
					+ 2.0 * dsf_alpha * tf.exp(-tf.square(dsf_alpha * cutoff_dist)) / (tf.sqrt(np.pi) * cutoff_dist))
	dsf_potential = tf.where(tf.greater(dists, cutoff_dist), tf.zeros_like(dsf_potential), dsf_potential)
	if return_grad:
		dsf_gradient = -(tf.erfc(dsf_alpha * dists) / tf.square(dists) - tf.erfc(dsf_alpha * cutoff_dist) / tf.square(cutoff_dist) \
	 					+ 2.0 * dsf_alpha / tf.sqrt(np.pi) * (tf.exp(-tf.square(dsf_alpha * dists)) / dists \
						- tf.exp(-tf.square(dsf_alpha * cutoff_dist)) / tf.sqrt(np.pi)))
		return dsf_potential, dsf_gradient
	else:
		return dsf_potential

def gs_canonicalize(dxyzs, nearest_neighbors):
	case_indices = tf.range(0, tf.shape(dxyzs)[0])
	first_axis = tf.gather_nd(dxyzs, tf.stack([case_indices, nearest_neighbors[:,0]], axis=1))
	first_axis /= tf.norm(first_axis, axis=-1, keep_dims=True)
	second_axis = tf.gather_nd(dxyzs, tf.stack([case_indices, nearest_neighbors[:,1]], axis=1))
	second_axis -= tf.expand_dims(tf.einsum('ij,ij->i',first_axis, second_axis), axis=-1) * first_axis
	second_axis /= tf.norm(second_axis, axis=-1, keep_dims=True)
	third_axis = tf.cross(first_axis, second_axis)
	transform_matrix = tf.stack([first_axis, second_axis, third_axis], axis=1)
	canon_xyzs = tf.einsum("lij,lkj->lki", transform_matrix, dxyzs)

	first_axis = tf.gather_nd(dxyzs, tf.stack([case_indices, nearest_neighbors[:,1]], axis=1))
	first_axis /= tf.norm(first_axis, axis=-1, keep_dims=True)
	second_axis = tf.gather_nd(dxyzs, tf.stack([case_indices, nearest_neighbors[:,0]], axis=1))
	second_axis -= tf.expand_dims(tf.einsum('ij,ij->i',first_axis, second_axis), axis=-1) * first_axis
	second_axis /= tf.norm(second_axis, axis=-1, keep_dims=True)
	third_axis = tf.cross(first_axis, second_axis)
	transform_matrix = tf.stack([first_axis, second_axis, third_axis], axis=1)
	perm_canon_xyzs = tf.einsum("lij,lkj->lki", transform_matrix, dxyzs)
	return canon_xyzs, perm_canon_xyzs


def center_dxyzs(xyzs, Zs):
	padding_mask = tf.where(tf.not_equal(Zs, 0))
	central_atom_coords = tf.gather_nd(xyzs, padding_mask)
	mol_coords = tf.gather(xyzs, padding_mask[:,0])
	dxyzs = tf.expand_dims(central_atom_coords, axis=1) - mol_coords
	Z_product = tf.expand_dims(tf.gather_nd(Zs, padding_mask), axis=1) * tf.gather(Zs, padding_mask[:,0])
	mask = tf.expand_dims(tf.where(tf.not_equal(Z_product, 0), tf.ones_like(Z_product, dtype=eval(PARAMS["tf_prec"])),
		tf.zeros_like(Z_product, dtype=eval(PARAMS["tf_prec"]))), axis=-1)
	dxyzs *= mask
	return dxyzs, padding_mask


def sparsify_coords(xyzs, Zs, pairs):
	padding_mask = tf.where(tf.not_equal(Zs, 0))
	central_atom_coords = tf.gather_nd(xyzs, padding_mask)
	pairs = tf.gather_nd(pairs, padding_mask)
	pair_mask = tf.where(tf.equal(pairs[...,-1], 0), tf.zeros_like(pairs[...,-1], dtype=eval(PARAMS["tf_prec"])), tf.ones_like(pairs[...,-1], dtype=eval(PARAMS["tf_prec"])))
	pair_coords = tf.gather_nd(xyzs, pairs[...,:-1])
	pair_Zs = pairs[...,-1]
	dxyzs = (tf.expand_dims(central_atom_coords, axis=1) - pair_coords) * tf.expand_dims(pair_mask, axis=-1)
	return dxyzs, pair_Zs
