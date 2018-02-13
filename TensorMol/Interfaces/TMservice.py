"""
A Socketed (SocketsIO) calculation server
Ideally this could be a prototype for similar Psi4 and PySCF.
"""

import sys, argparse, json, time
import numpy as np
import socketIO_client
from socketIO_client import SocketIO, LoggingNamespace
from TensorMol import *
parser = argparse.ArgumentParser()
parser.add_argument('--port', help='port')
results, remaining = parser.parse_known_args()

PORT = int(results.port) if results.port is not None else 31415

def wake():
	a=MSet()
	m=Mol()
	m.FromXYZString("""4

	C 1. 0. 0.
	H 0. 1. 0.
	N 0. 0. 1.
	O 1. 1. 0.""")
	a.mols.append(m)
	TreatedAtoms = np.array([1,6,7,8], dtype=np.uint8)
	# PARAMS["networks_directory"] = "/home/animal/Packages/TensorMol/networks/"
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["NeuronType"] = "sigmoid_with_param"
	PARAMS["sigmoid_alpha"] = 100.0
	PARAMS["HiddenLayers"] = [2000, 2000, 2000]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	PARAMS["EECutoffOff"] = 15.0
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP_Direct_EE_WithEle_Release(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	PARAMS["DSFAlpha"] = 0.18*BOHRPERA
	manager=TFMolManage("chemspider12_solvation", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	return manager

def EnAndForce(m_,mgr_):
	Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = mgr_.EvalBPDirectEEUpdateSingle(m_, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
	energy = Etotal
	return energy,-1.0*gradient

tm = wake()
# Run a quick test and get evalprepare called.
m=Mol()
m.FromXYZString("""4

C 1. 0. 0.
H 0. 1. 0.
N 0. 0. 1.
O 1. 1. 0.""")
EnAndForce(m,tm)

#
# This is the base which should be inherited by PYSCF, PSI4 etc.
#
class CalcService:
	def __init__(self):
		import logging
		logging.getLogger('socketIO-client').setLevel(logging.DEBUG)
		logging.basicConfig()
		self.socketIO = SocketIO('localhost', PORT,LoggingNamespace)
		self.socketIO.on('connect', self.on_connect)
		self.socketIO.on('disconnect', self.on_disconnect)
		self.socketIO.on('reconnect', self.on_reconnect)
		self.socketIO.on('newjob', self.on_newjob)
		self.socketIO.wait()
		return
	def on_connect(self):
		LOGGER.debug('Py:connect')
		#self.socketIO.emit('newresult','hi')
	def on_disconnect(self):
		LOGGER.debug('Py:disconnect')
		exit(0)
	def on_reconnect(self):
		LOGGER.debug('Py:reconnect')
	def emitResult(self,calcid_,m_,energy_=0.0,force_=None):
		"""
		This routine needs to be generalized to return anything we could want.
		"""
		geom = ''.join((str(m_)).split()[2:])
		to_emit={"calculation_id":calcid_,"geometry":geom,"energy":energy_,"force":force_}
		LOGGER.debug('Emitting back')
		self.socketIO.emit('newresult',to_emit)
		return
	def emitBatch(self,calcid_,m_,energy_=0.0,force_=None):
		"""
		This routine needs to be generalized to return anything we could want.
		"""
		geom = ''.join((str(m_)).split()[2:])
		to_emit={"calculation_id":calcid_,"geometry":geom,"energy":energy_,"force":force_}
		LOGGER.debug('Emitting back')
		self.socketIO.emit('newresult',to_emit)
		return
	def on_newjob(self,*args,**kwargs):
		if (len(args)<3):
			return
		cid = args[0]
		ctype = args[1]
		coords = args[2]
		# At this point we should really route this to different
		# Types of calculation handlers.
		m = Mol()
		scoords = str(len(coords.split('\n')))+"\n"+"\n"+coords
		print("----\n"+scoords+"\n----\n")
		m.FromXYZString(scoords)
		def EnAndForce(x_, DoForce=True):
			mtmp = Mol(m.atoms,x_)
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = tm.EvalBPDirectEEUpdateSingle(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal
			force = gradient
			if DoForce:
				return energy, force
			else:
				return energy
		energy, force = EnAndForce(m.coords)
		if (ctype=="geometryOptimization"):
			Opt = GeomOptimizer(EnAndForce)
			m = Opt.Opt(m)
		#elif (ctype=="configSearch"):
		#	Opt = GeomOptimizer(EnAndForce)
		#	m = Opt.Opt(m)
		self.emitResult(cid,m,energy)
#start process

CalcService()
