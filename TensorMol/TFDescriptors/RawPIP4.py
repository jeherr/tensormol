"""
An encoded version of a 4-body
permutationally invariant polynomial.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_functiont function
if (HAS_TF):
	import tensorflow as tf

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

def Pip2(xyzs_,Zs_,atom_codes,doubs_):
	"""
	Args
	"""
	return

def Pip3(xyzs_,Zs_,atom_codes,trips_):
	"""
	Args
	"""
	return

def Pip4(xyzs_,Zs_,atom_codes,quads_):
	"""
	Args
	"""
	return
