#include <Python.h>
#include <numpy/arrayobject.h>
#include <dictobject.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include "SH.hpp"

#if PY_MAJOR_VERSION >= 3
    #define PyInt_FromLong PyLong_FromLong
		#define PyInt_AS_LONG PyLong_AS_LONG
#endif

// So that parameters can be dealt with elsewhere.
static SHParams ParseParams(PyObject *Pdict)
{
	SHParams tore;
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "RBFS");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.RBFS = (double*)RBFa->data;
	}
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "SRBF");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.SRBF = (double*)RBFa->data;
	}
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "ANES");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.ANES = (double*)RBFa->data;
	}
	tore.SH_LMAX = (int)PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_LMAX")));
	tore.SH_NRAD = (int)PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_NRAD")));
	tore.SH_ORTH = (int)PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_ORTH")));
	tore.SH_MAXNR = (int)PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_MAXNR")));
	// HACK TO DEBUG>>>>>
	/*
	double t[9] = {1.0,0.,0.,0.,1.,0.,0.,0.,1.};
	double* out;
	TransInSHBasis(&tore,t, out);
	*/
	return tore;
}

static SymParams ParseSymParams(PyObject *Pdict)
{
	SymParams tore;
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "AN1_r_Rs");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.r_Rs = (double*)RBFa->data;
	}
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "AN1_a_Rs");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.a_Rs = (double*)RBFa->data;
	}
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "AN1_a_As");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.a_As = (double*)RBFa->data;
	}
	tore.num_r_Rs = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"AN1_num_r_Rs")));
	tore.num_a_Rs = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"AN1_num_a_Rs")));
	tore.num_a_As = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"AN1_num_a_As")));
	tore.r_Rc = PyFloat_AsDouble((PyDict_GetItemString(Pdict,"AN1_r_Rc")));
	tore.a_Rc = PyFloat_AsDouble((PyDict_GetItemString(Pdict,"AN1_a_Rc")));
	tore.eta = PyFloat_AsDouble((PyDict_GetItemString(Pdict,"AN1_eta")));
	tore.zeta = PyFloat_AsDouble((PyDict_GetItemString(Pdict,"AN1_zeta")));
	return tore;
}

inline double fc(const double &dist, const double &dist_cut) {
	if (dist > dist_cut)
		return(0.0);
	else
		return (0.5*(cos(PI*dist/dist_cut)+1));
};

inline double gaussian(const double dist_cut, const int ngrids, double dist, const int j,  const double  width, const double height)  {
	double position;
	position = dist_cut / (double)ngrids * (double)j;
	return height*exp(-((dist - position)*(dist-position))/(2*width*width));
};

struct MyComparator
{
	const std::vector<double> & value_vector;

	MyComparator(const std::vector<double> & val_vec):
	value_vector(val_vec) {}

	bool operator()(int i1, int i2)
	{
		return value_vector[i1] < value_vector[i2];
	}
};


double dist(double x0,double y0,double z0,double x1,double y1,double z1)
{
	return sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1));
}

//
//   Everything below here is exposed to python
//

// To check all your python library is linked correctly.
static PyObject* EmptyInterfacedFunction(PyObject *self, PyObject  *args)
{
	int To;
	PyArrayObject *xyz;
	std::cout << "Parsing..." << endl;
	try {if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &xyz, &To))
		return NULL;}
		catch(const std::exception &exc)
	{
		std::cout << exc.what();
	}
	std::cout << "Parsed..." << endl;
		return Py_BuildValue("i", To);
	}

static PyObject* Make_SH(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut;
	int  ngrids,  theatom;
	PyObject *Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!i", &PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &atoms_ , &theatom))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	uint8_t* atoms=(uint8_t*)atoms_->data;
	npy_intp* Nxyz = xyz->dimensions; // Now assumed to be natomX3.
	int natom;
	natom = Nxyz[0];

	//npy_intp outdim[2] = {natom,SH_NRAD*(1+SH_LMAX)*(1+SH_LMAX)};
	npy_intp outdim[2] = {1,Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX)};
	if (theatom<0)
	outdim[0] = natom;
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data, *grids_data;
	double center[3]; // x y z of the center
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	// Note that Grids are currently unused...
	//grids_data = (double*) grids -> data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	int SHdim = SizeOfGauSH(Prm);
	if (theatom<0)
	{
		#pragma omp parallel for
		for (int i=0; i<natom; ++i)
		{
			double xc = xyz_data[i*3];
			double yc = xyz_data[i*3+1];
			double zc = xyz_data[i*3+2];
			for (int j = 0; j < natom; j++)
			{
				double x = xyz_data[j*3];
				double y = xyz_data[j*3+1];
				double z = xyz_data[j*3+2];
				//RadSHProjection(x-xc,y-yc,z-zc,SH_data + i*SH_NRAD*(1+SH_LMAX)*(1+SH_LMAX), natom);
				//RadSHProjection(Prm,x-xc,y-yc,z-zc,SH_data + i*SHdim, (double)atoms[j]);
				RadSHProjection(Prm,x-xc,y-yc,z-zc,SH_data + i*SHdim, Prm->ANES[atoms[j]-1]);
			}
		}
	}
	else{
		int i = theatom;
		int ai=0;


		double xc = xyz_data[i*3];
		double yc = xyz_data[i*3+1];
		double zc = xyz_data[i*3+2];

		for (int j = 0; j < natom; j++)
		{
			double x = xyz_data[j*3];
			double y = xyz_data[j*3+1];
			double z = xyz_data[j*3+2];
			RadSHProjection(Prm,x-xc,y-yc,z-zc,SH_data + ai*SHdim, Prm->ANES[atoms[j]-1]);
		}
	}
	return SH;
}

//
// Embed molecule using Gaussian X Spherical Harmonic basis
// Does all atoms if theatom argument < 0
// These vectors are flattened [gau,l,m] arrays
// This version embeds a single atom after several transformations.
//
static PyObject* Make_SH_Transf(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_, *transfs;
	double   dist_cut;
	int  theatom;
	PyObject *Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!iO!", &PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &atoms_ , &theatom, &PyArray_Type, &transfs ))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	uint8_t* atoms=(uint8_t*)atoms_->data;

	npy_intp* Nxyz = xyz->dimensions; // Now assumed to be natomX3.
	npy_intp* Ntrans = transfs->dimensions; // Now assumed to be natomX3.
	int natom, ntr;

	natom = Nxyz[0];
	ntr = Ntrans[0];

	npy_intp outdim[2] = {ntr,Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX)};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data, *grids_data, *t_data;
	double center[3]; // x y z of the center
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	t_data = (double*) ((PyArrayObject*) transfs)->data;
	// Note that Grids are currently unused...
	//grids_data = (double*) grids -> data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	double t_coords0[natom][3];
	int SHdim = SizeOfGauSH(Prm);

	{
		// Center the atom and then perform the transformations.

		double xc = xyz_data[theatom*3];
		double yc = xyz_data[theatom*3+1];
		double zc = xyz_data[theatom*3+2];

		for (int j = 0; j < natom; j++)
		{
			t_coords0[j][0] = xyz_data[j*3] - xc;
			t_coords0[j][1] = xyz_data[j*3+1] - yc;
			t_coords0[j][2] = xyz_data[j*3+2] - zc;
		}

		#pragma omp parallel for
		for (int i=0; i<ntr; ++i)
		{
			double* tr = t_data+i*9;
			/*
			cout << tr[0] << tr[1] << tr[2] << endl;
			cout << tr[3] << tr[4] << tr[5] << endl;
			cout << tr[6] << tr[7] << tr[8] << endl;
			*/
			for (int j = 0; j < natom; j++)
			{
				// Perform the transformation, embed and out...
				double x = (tr[0*3]*t_coords0[j][0]+tr[0*3+1]*t_coords0[j][1]+tr[0*3+2]*t_coords0[j][2]);
				double y = (tr[1*3]*t_coords0[j][0]+tr[1*3+1]*t_coords0[j][1]+tr[1*3+2]*t_coords0[j][2]);
				double z = (tr[2*3]*t_coords0[j][0]+tr[2*3+1]*t_coords0[j][1]+tr[2*3+2]*t_coords0[j][2]);
				RadSHProjection(Prm, x, y, z, SH_data + i*SHdim, Prm->ANES[atoms[j]-1]);
			}
		}
	}
	return SH;
}


//
// Embed molecule using Gaussian X Spherical Harmonic basis
// Does all atoms... just for debug really.
// Make invariant embedding of all the atoms.
// This is not a reversible embedding.
static PyObject* Make_Inv(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut,  mask, mask_prob,dist;
	int theatom;
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!i",
	&PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &atoms_ , &theatom))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;
	npy_intp* Nxyz = xyz->dimensions;
	int natom, num_CM;
	natom = Nxyz[0];

	npy_intp outdim[2];
	if (theatom>=0)
	outdim[0] = 1;
	else
	outdim[0] = natom;
	outdim[1]=Prm->SH_NRAD*(1+Prm->SH_LMAX);
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data, *grids_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	//grids_data = (double*) grids -> data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;

	/*double center[3]={0.,0.,0.};
	for (int j = 0; j < natom; j++)
	{
	center[0]+=xyz_data[j*Nxyz[1]+0];
	center[1]+=xyz_data[j*Nxyz[1]+1];
	center[2]+=xyz_data[j*Nxyz[1]+2];
}
center[0]/= natom;
center[1]/= natom;
center[2]/= natom;*/

if (theatom >= 0)
{
	int i = theatom;
	double xc = xyz_data[i*Nxyz[1]+0];
	double yc = xyz_data[i*Nxyz[1]+1];
	double zc = xyz_data[i*Nxyz[1]+2];
	for (int j = 0; j < natom; j++)
	{
		double x = xyz_data[j*Nxyz[1]+0];
		double y = xyz_data[j*Nxyz[1]+1];
		double z = xyz_data[j*Nxyz[1]+2];
		RadInvProjection(Prm, x-xc,y-yc,z-zc,SH_data,(double)atoms[j]);
	}
}
else
{
	#pragma omp parallel for
	for (int i=0; i<natom; ++i )
	{
		double xc = xyz_data[i*Nxyz[1]+0];
		double yc = xyz_data[i*Nxyz[1]+1];
		double zc = xyz_data[i*Nxyz[1]+2];
		for (int j = 0; j < natom; j++)
		{
			double x = xyz_data[j*Nxyz[1]+0];
			double y = xyz_data[j*Nxyz[1]+1];
			double z = xyz_data[j*Nxyz[1]+2];
			RadInvProjection(Prm, x-xc,y-yc,z-zc,SH_data+i*(outdim[1]),(double)atoms[j]);
		}
	}
}
return SH;
}

//
// returns a [Nrad X Nang] x [npts] array which contains rasterized versions of the
// non-orthogonal basis functions.
//
static PyObject* Raster_SH(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut,  mask, mask_prob,dist;
	int theatom;
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!O!", &PyDict_Type, &Prm_, &PyArray_Type, &xyz))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	const int npts = (xyz->dimensions)[0];
	int nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	npy_intp outdim[2] = {nbas,npts};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data;
	double center[3]; // x y z of the center
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;

	for (int pt=0; pt<npts; ++pt)
	{
		double r = sqrt(xyz_data[pt*3+0]*xyz_data[pt*3+0]+xyz_data[pt*3+1]*xyz_data[pt*3+1]+xyz_data[pt*3+2]*xyz_data[pt*3+2]);
		double theta = acos(xyz_data[pt*3+2]/r);
		double phi = atan2(xyz_data[pt*3+1],xyz_data[pt*3+0]);

		//		cout << "r" << r << endl;
		//		cout << "r" << theta << endl;
		//		cout << "r" << phi << endl;

		int bi = 0;
		for (int i=0; i<Prm->SH_NRAD ; ++i)
		{
			double Gv = Gau(r, Prm->RBFS[i*2],Prm->RBFS[i*2+1]);
			for (int l=0; l<Prm->SH_LMAX+1 ; ++l)
			{
				//				cout << "l=" << l << " Gv "<< Gv <<endl;
				for (int m=-l; m<l+1 ; ++m)
				{
					SH_data[bi*npts+pt] += Gv*RealSphericalHarmonic(l,m,theta,phi);
					//					cout << SH_data[bi*npts+pt] << endl;
					++bi;
				}
			}
		}
	}
	return SH;
}

// Gives the projection of a delta function at xyz
static PyObject* Project_SH(PyObject *self, PyObject  *args)
{
	double x,y,z;
	PyArrayObject *xyz;
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!O!", &PyDict_Type, &Prm_, &PyArray_Type, &xyz))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	int nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	npy_intp outdim[2] = {1,nbas};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data,*xyz_data;

	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	x=xyz_data[0];
	y=xyz_data[1];
	z=xyz_data[2];
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	int Nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	//	int Nang = (1+SH_LMAX)*(1+SH_LMAX);
	double r = sqrt(x*x+y*y+z*z);

	if (r<pow(10.0,-11.0))
	return SH;
	double theta = acos(z/r);
	double phi = atan2(y,x);

	//	cout << "R,theta,phi" << r << " " << theta << " " <<phi << " " <<endl;
	int bi = 0;
	for (int i=0; i<Prm->SH_NRAD ; ++i)
	{
		double Gv = Gau(r, Prm->RBFS[i*2],Prm->RBFS[i*2+1]);
		cout << Prm->RBFS[i*2] << " " << Gv << endl;

		for (int l=0; l<Prm->SH_LMAX+1 ; ++l)
		{
			for (int m=-l; m<l+1 ; ++m)
			{
				SH_data[bi] += Gv*RealSphericalHarmonic(l,m,theta,phi);
				++bi;
			}
		}
	}
	return SH;
}

static PyObject* Make_DistMat(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &xyz))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {nat,nat};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data,*xyz_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	for (int i=0; i < nat; ++i)
	for (int j=i+1; j < nat; ++j)
	{
		SH_data[i*nat+j] = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2])) + 0.00000000001;
		SH_data[j*nat+i] = SH_data[i*nat+j];
	}
	return SH;
}

static PyObject* Make_DistMat_ForReal(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	int nreal;
	if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &xyz, &nreal))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {nreal,nat};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data,*xyz_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	for (int i=0; i < nreal; ++i)
	for (int j=0; j < nat; ++j)
	{
		SH_data[i*nat+j] = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2])) + 0.00000000001;
	}
	return SH;
}

/* counts the number of atoms which occur within a radius of those of type z1*/
static PyObject* CountInRange(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	PyArrayObject *Zs;
	double cut;
	double dr;
	int nreal, ele1, ele2;
	if (!PyArg_ParseTuple(args, "O!O!iiidd", &PyArray_Type, &Zs, &PyArray_Type, &xyz,  &nreal, &ele1, &ele2, &cut, &dr))
		return NULL;
	const int nat = (xyz->dimensions)[0];
	double *SH_data, *xyz_data;
	int outdim = int(cut/dr);
	npy_intp outdima[1] = {outdim};
	PyObject* SH = PyArray_ZEROS(1, outdima, NPY_DOUBLE, 0);
	uint8_t* atoms=(uint8_t*)Zs->data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	double dist;
	double nav = 0.0;
	int di = 0;
	for (int i=0; i < nreal; ++i)
		if (atoms[i] == ele1) {
			nav += 1.0;
			for (int j=0; j < nat; ++j)
			{
				if (atoms[j] == ele2 && i != j) {
					dist = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2]));
					di = int(dist/dr);
					if (di <outdim)
						for (int k=di; k<outdim; ++k)
							SH_data[k] += 1.0;
				}
			}
		}
	for (int k=0; k<outdim; ++k)
		SH_data[k] /= nav;
	return SH;
}

//
// Make a neighborlist using a naive, quadratic algorithm.
// returns a python list.
// Only does up to the nreal-th atom. (for periodic tesselation)
//
static PyObject* Make_NListNaive(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	double rng;
	int nreal;
	int DoPerms;
	if (!PyArg_ParseTuple(args, "O!dii", &PyArray_Type, &xyz, &rng, &nreal, &DoPerms))
		return NULL;
	double *xyz_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	const int nat = (xyz->dimensions)[0];
	// Avoid stupid python reference counting issues by just using std::vector...
	// Argsort by x.
	std::vector<double> XX;
	XX.assign(xyz_data,xyz_data+3*nat);
	std::vector<int> y(nat);
	std::size_t n(0);
	std::generate(y.begin(), y.end(), [&]{ return n++; });
	std::sort(y.begin(),y.end(), [&](int i1, int i2) { return XX[i1*3] < XX[i2*3]; } );
	// So y now contains sorted x indices, do the skipping Neighbor list.
	std::vector< std::vector<int> > tmp(nreal);
	for (int i=0; i< nat; ++i)
	{
		int I = y[i];
			// We always work in order of increasing X...
			for (int j=i+1; j < nat; ++j)
			{
				int J = y[j];
				if (!(I<nreal || J<nreal))
					continue;
				if (fabs(XX[I*3] - XX[J*3]) > rng)//((XX[J*3] - XX[I*3]) > rng)//
				{
					break;
				}
				double dx = (xyz_data[I*3+0]-xyz_data[J*3+0]);
				double dy = (xyz_data[I*3+1]-xyz_data[J*3+1]);
				double dz = (xyz_data[I*3+2]-xyz_data[J*3+2]);
				double dij = sqrt(dx*dx+dy*dy+dz*dz) + 0.0000000000001;
				if (dij < rng)
				{
					if (I<J)
					{
						tmp[I].push_back(J);
						if (J<nreal && DoPerms==1)
							tmp[J].push_back(I);
					}
					else
					{
						tmp[J].push_back(I);
						if (I<nreal && DoPerms==1)
							tmp[I].push_back(J);
					}
				}
			}
	}
	PyObject* Tore = PyList_New(nreal);
	for (int i=0; i < nreal; ++i)
	{
		PyObject* tl = PyList_New(tmp[i].size());
		std::sort(tmp[i].begin(),tmp[i].end());
		for (int j=0; j<tmp[i].size();++j)
		{
			PyObject* ti = PyInt_FromLong(tmp[i][j]);
			PyList_SetItem(tl,j,ti);
		}
		PyList_SetItem(Tore,i,tl);
	}
	return Tore;
}

//
// Like the above, but returns a -1 padded Tensor with sorted Neighbor inds.
// And works batchwise on a Nmol X MaxNatom X 3 distance tensor.
//
static PyObject* Make_NLTensor(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyzs;
	PyArrayObject *zs;
	double rng;
	int nreal;
	int DoPerms;
	if (!PyArg_ParseTuple(args, "O!O!dii", &PyArray_Type, &xyzs, &PyArray_Type, &zs, &rng, &nreal, &DoPerms))
		return NULL;
	double *xyzs_data;
	int32_t *z_data;
	xyzs_data = (double*) ((PyArrayObject*) xyzs)->data;
	z_data = (int32_t*) ((PyArrayObject*) zs)->data;
	const int nmol = (xyzs->dimensions)[0];
	const int nat = (xyzs->dimensions)[1];

	struct ind_dist {
		int ind;
		double dist;
	};
	struct by_dist {
			bool operator()(ind_dist const &a, ind_dist const &b) {
				if (a.dist > 0.01 && b.dist > 0.01)
					return a.dist < b.dist;
				else if (a.dist > 0.01)
					return true;
				else
					return false;
			}
	};

	typedef std::vector< std::vector<ind_dist> > vov;
	typedef std::vector< vov > vovov;
	vovov NLS;

	for (int k=0; k<nmol; ++k)
	{
		double* xyz_data = xyzs_data+k*(3*nat);
		std::vector<double> XX;
		XX.assign(xyz_data,xyz_data+3*nat);
		std::vector<int> y(nat);
		std::size_t n(0);
		std::generate(y.begin(), y.end(), [&]{ return n++; });
		std::sort(y.begin(),y.end(), [&](int i1, int i2) { return XX[i1*3] < XX[i2*3]; } );
		// So y now contains sorted x indices, do the skipping Neighbor list.
		vov tmp(nreal);
		for (int i=0; i< nat; ++i)
		{
			int I = y[i];
			// We always work in order of increasing X...
			for (int j=i+1; j < nat; ++j)
			{
				int J = y[j];
				if (!(I<nreal || J<nreal))
					continue;

				if (z_data[k*(nat)+I]<=0 || z_data[k*(nat)+J]<=0)
					continue;
				if (fabs(XX[I*3] - XX[J*3]) > rng)
					break;

				double dx = (xyz_data[I*3+0]-xyz_data[J*3+0]);
				double dy = (xyz_data[I*3+1]-xyz_data[J*3+1]);
				double dz = (xyz_data[I*3+2]-xyz_data[J*3+2]);
				double dij = sqrt(dx*dx+dy*dy+dz*dz) + 0.0000000000001;
				if (dij < rng)
				{
					ind_dist Id = {I,dij};
					ind_dist Jd = {J,dij};
					if (I<J)
					{
						tmp[I].push_back(Jd);
						if (J<nreal && DoPerms==1)
							tmp[J].push_back(Id);
					}
					else
					{
						tmp[J].push_back(Id);
						if (I<nreal && DoPerms==1)
							tmp[I].push_back(Jd);
					}
				}
			}
		}
		NLS.push_back(tmp);
	}
	// Determine the maximum number of neighbors and make a tensor.
	int MaxNeigh = 0;

	for (int i = 0; i<NLS.size(); ++i)
	{
		vov& tmp = NLS[i];
		for (int j = 0; j<tmp.size(); ++j)
		{
			if (tmp[j].size() > MaxNeigh)
				MaxNeigh = tmp[j].size();
			std::sort(tmp[j].begin(), tmp[j].end(), by_dist());
		}
	}
	npy_intp outdim2[3] = {nmol,nat,MaxNeigh};
	PyObject* NLTensor = PyArray_ZEROS(3, outdim2, NPY_INT32,0);
	int32_t* NL_data = (int32_t*) ((PyArrayObject*)NLTensor)->data;
	for (int i = 0; i<nmol; ++i)
	{
		for (int j = 0; j<nat; ++j)
		{
			for (int k=0; k<MaxNeigh; ++k)
			{
				if (k < NLS[i][j].size())
					NL_data[i*(nat*MaxNeigh)+j*MaxNeigh+k] = (int32_t)(NLS[i][j][k].ind);
				else
					NL_data[i*(nat*MaxNeigh)+j*MaxNeigh+k] = (int32_t)(-1);
			}
		}
	}
	return NLTensor;
}


//
// Makes a triples tensor.
// if DoPerms = True
// Output is nMol X MaxNAtom X (MaxNeigh * MaxNeigh-1) X 2
// else:
// Output is nMol X MaxNAtom X (MaxNeigh * ((MaxN+1)/2-1)) X 2
static PyObject* Make_TLTensor(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyzs;
	PyArrayObject *zs;
	double rng;
	int nreal;
	int DoPerms;
	if (!PyArg_ParseTuple(args, "O!O!dii", &PyArray_Type, &xyzs, &PyArray_Type, &zs, &rng, &nreal, &DoPerms))
		return NULL;
	double *xyzs_data;
	int32_t *z_data;
	xyzs_data = (double*) ((PyArrayObject*) xyzs)->data;
	z_data = (int32_t*) ((PyArrayObject*) zs)->data;
	const int nmol = (xyzs->dimensions)[0];
	const int nat = (xyzs->dimensions)[1];

	struct ind_dist {
		int ind;
		double dist;
	};
	struct by_dist {
			bool operator()(ind_dist const &a, ind_dist const &b) {
				if (a.dist > 0.01 && b.dist > 0.01)
					return a.dist < b.dist;
				else if (a.dist > 0.01)
					return true;
				else
					return false;
			}
	};

	typedef std::vector< std::vector<ind_dist> > vov;
	typedef std::vector< vov > vovov;
	vovov NLS;

	for (int k=0; k<nmol; ++k)
	{
		double* xyz_data = xyzs_data+k*(3*nat);
		std::vector<double> XX;
		XX.assign(xyz_data,xyz_data+3*nat);
		std::vector<int> y(nat);
		std::size_t n(0);
		std::generate(y.begin(), y.end(), [&]{ return n++; });
		std::sort(y.begin(),y.end(), [&](int i1, int i2) { return XX[i1*3] < XX[i2*3]; } );
		// So y now contains sorted x indices, do the skipping Neighbor list.
		vov tmp(nreal);
		for (int i=0; i< nat; ++i)
		{
			int I = y[i];
			// We always work in order of increasing X...
			for (int j=i+1; j < nat; ++j)
			{
				int J = y[j];
				if (!(I<nreal || J<nreal))
					continue;

				if (z_data[k*(nat)+I]<=0 || z_data[k*(nat)+J]<=0)
					continue;
				if (fabs(XX[I*3] - XX[J*3]) > rng)
					break;

				double dx = (xyz_data[I*3+0]-xyz_data[J*3+0]);
				double dy = (xyz_data[I*3+1]-xyz_data[J*3+1]);
				double dz = (xyz_data[I*3+2]-xyz_data[J*3+2]);
				double dij = sqrt(dx*dx+dy*dy+dz*dz) + 0.0000000000001;
				if (dij < rng)
				{
					ind_dist Id = {I,dij};
					ind_dist Jd = {J,dij};
					if (I<J)
					{
						tmp[I].push_back(Jd);
						if (J<nreal && DoPerms==1)
							tmp[J].push_back(Id);
					}
					else
					{
						tmp[J].push_back(Id);
						if (I<nreal && DoPerms==1)
							tmp[I].push_back(Jd);
					}
				}
			}
		}
		NLS.push_back(tmp);
	}
	// Determine the maximum number of neighbors and make a tensor.
	int MaxNeigh = 0;
	for (int i = 0; i<NLS.size(); ++i)
	{
		vov& tmp = NLS[i];
		for (int j = 0; j<tmp.size(); ++j)
		{
			if (tmp[j].size() > MaxNeigh)
				MaxNeigh = tmp[j].size();
			std::sort(tmp[j].begin(), tmp[j].end(), by_dist());
		}
	}

	int Dim2 = MaxNeigh*(MaxNeigh-1);
	if (!DoPerms)
		Dim2 = MaxNeigh*(MaxNeigh+1)/2 - MaxNeigh;

	npy_intp outdim2[4] = {nmol,nat,Dim2,2};
	PyObject* NLTensor = PyArray_ZEROS(4, outdim2, NPY_INT32,0);
	int32_t* NL_data = (int32_t*) ((PyArrayObject*)NLTensor)->data;
	memset(NL_data, -1, sizeof(int32_t)*nmol*nat*Dim2*2);
	for (int i = 0; i<nmol; ++i)
	{
		for (int j = 0; j<nat; ++j)
		{
			int counter = 0;
			for (int k=0; k<MaxNeigh; ++k)
			{
				for (int l=0; l<MaxNeigh; ++l)
				{
					if (k < NLS[i][j].size())
					{
						if (DoPerms && k!=l)
						{
							NL_data[i*(nat*Dim2*2)+j*(Dim2*2)+counter*2] = (int32_t)(NLS[i][j][k].ind);
							NL_data[i*(nat*Dim2*2)+j*(Dim2*2)+counter*2+1] = (int32_t)(NLS[i][j][l].ind);
							counter++;
						}
						else if (k<l)
						{
							NL_data[i*(nat*Dim2*2)+j*(Dim2*2)+counter*2] = (int32_t)(NLS[i][j][k].ind);
							NL_data[i*(nat*Dim2*2)+j*(Dim2*2)+counter*2+1] = (int32_t)(NLS[i][j][l].ind);
							counter++;
						}
					}
				}
			}
		}
	}
	return NLTensor;
}


//
// Linear scaling version of the above routine.
// This should NOT be used for training.
//
static PyObject* Make_NListLinear(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	double Rc;
	double SkinDepth = 0.0;
	// This is used to avoid recomputation.
	int nreal;
	int DoPerms;
	if (!PyArg_ParseTuple(args, "O!dii", &PyArray_Type, &xyz, &Rc, &nreal,&DoPerms))
		return NULL;
	double *x;
	x = (double*) ((PyArrayObject*) xyz)->data;
	const int nat = (xyz->dimensions)[0];
	const int nx = nat;
	double xmx = x[0];
	double xmn = x[0];
	double ymx = x[1];
	double ymn = x[1];
	double zmx = x[2];
	double zmn = x[2];
	for (int II = 0; II<nat ; ++II )
	{
		if (x[II*3]<xmn)
			xmn = x[II*3];
		if (x[II*3]>xmx)
			xmx = x[II*3];
		if (x[II*3+1]<ymn)
			ymn = x[II*3+1];
		if (x[II*3+1]>ymx)
			ymx = x[II*3+1];
		if (x[II*3+2]<zmn)
			zmn = x[II*3+2];
		if (x[II*3+2]>zmx)
			zmx = x[II*3+2];
	}
	double lx = xmx-xmn;
	double ly = ymx-ymn;
	double lz = zmx-zmn;
	double dx = lx/Rc;
	double dy = ly/Rc;
	double dz = lz/Rc;
	//double rho = nx/(lx*ly*lz);
	// divide space into overlapping cubic prisms of size
	// Rc+SkinDepth in raster order.
	int NII = (int(lx/dx)+1);
	int NJJ = (int(ly/dy)+1);
	int NKK = (int(lz/dz)+1);
	int NCubes = NII*NJJ*NKK;
	int ndist = 0;
	std::vector< std::vector<int> > tmp(nreal);
	for (int II=0; II < NII ; ++II)
	{
		for (int JJ=0; JJ < NJJ ; ++JJ)
		{
			for (int KK=0; KK < NKK ; ++KK)
			{
				int tmpcube[nx];
				int NInCube = 0;
				// The cube's center is at
				double Cx = xmn+dx*II;
				double Cy = ymn+dy*JJ;
				double Cz = zmn+dz*KK;
				double xlb =  Cx-Rc-SkinDepth;
				double xub =  Cx+Rc+SkinDepth;
				double ylb =  Cy-Rc-SkinDepth;
				double yub =  Cy+Rc+SkinDepth;
				double zlb =  Cz-Rc-SkinDepth;
				double zub =  Cz+Rc+SkinDepth;
				/*cout << xmn << " X " << xmx << endl;
				cout << ymn << " X " << ymx << endl;
				cout << zmn << " X " << zmx << endl;
				cout << xlb << " x " << xub << endl;
				cout << ylb << " x " << yub << endl;
				cout << zlb << " x " << zub << endl;*/
				// Now assign cube lists.
				for (int i=0; i<nx;++i)
				{
					if (xlb < x[i*3+0] && x[i*3+0] < xub)
					{
						if (ylb < x[i*3+1] && x[i*3+1] < yub)
						{
							if (zlb < x[i*3+2] && x[i*3+2] < zub)
							{
								tmpcube[NInCube] = i;
								NInCube++;
							}
						}
					}
				}
				for (int i=0; i<NInCube; ++i)
				{
					int I = tmpcube[i];
					if (I>=nreal)
						continue;
					for (int j=i+1; j<NInCube; ++j)
					{
						int J = tmpcube[j];
						if (std::count(tmp[I].begin(),tmp[I].end(),J)==0)
						{
							ndist++;
							double xx = (x[I*3+0]-x[J*3+0]);
							double yy = (x[I*3+1]-x[J*3+1]);
							double zz = (x[I*3+2]-x[J*3+2]);
							double dij = sqrt(xx*xx+yy*yy+zz*zz) + 0.00000000001;
							//dists[std::make_pair(i,y[j])] = dij;
							if (dij < Rc)
							{
								tmp[I].push_back(J);
								if (J<nreal && DoPerms==1)
									tmp[J].push_back(I);
							}
						}
					}
				}
			}
		}
	}
	// Avoid stupid python reference counting issues by just using std::vector...
	PyObject* Tore = PyList_New(nreal);
	for (int i=0; i < nreal; ++i)
	{
		PyObject* tl = PyList_New(tmp[i].size());
		for (int j=0; j<tmp[i].size();++j)
		{
			PyObject* ti = PyInt_FromLong(tmp[i][j]);
			PyList_SetItem(tl,j,ti);
		}
		PyList_SetItem(Tore,i,tl);
	}
	return Tore;
}

static PyObject* DipoleAutoCorr(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &xyz))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {nat,1};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data,*xyz_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
#pragma omp parallel for
	for (int i=0; i < nat; ++i) // Distance between points.
	{
		for (int j=0; j < nat-i; ++j) // points to sum over.
		{
			SH_data[i] += (xyz_data[j*3+0]*xyz_data[(j+i)*3+0]+xyz_data[j*3+1]*xyz_data[(j+i)*3+1]+xyz_data[j*3+2]*xyz_data[(j+i)*3+2]);
		}
		SH_data[i] /= double(nat-i);
	}
	return SH;
}

static PyObject* Norm_Matrices(PyObject *self, PyObject *args)
{
	PyArrayObject *dmat1, *dmat2;
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &dmat1, &PyArray_Type, &dmat2))
	return NULL;
	double norm = 0;
	const int dim1 = (dmat1->dimensions)[0];
	const int dim2 = (dmat1->dimensions)[1];
	double *dmat1_data, *dmat2_data;
	double normmat[dim1*dim2];
	dmat1_data = (double*) ((PyArrayObject*)dmat1)->data;
	dmat2_data = (double*) ((PyArrayObject*)dmat2)->data;
	#ifdef OPENMP
	#pragma omp parallel for reduction(+:norm)
	#endif
	for (int i=0; i < dim1; ++i)
	for (int j=0; j < dim2; ++j)
	norm += (dmat1_data[i*dim2+j] - dmat2_data[i*dim2+j])*(dmat1_data[i*dim2+j] - dmat2_data[i*dim2+j]);
	return PyFloat_FromDouble(sqrt(norm));
}

//
// returns a [Nrad X Nang] x [npts] array which contains rasterized versions of the
// non-orthogonal basis functions.
//
static PyObject* Overlap_SH(PyObject *self, PyObject  *args)
{
  //std::cout << "Recompiled... " << endl;
	//return NULL;
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &Prm_))
		return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	int Nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	npy_intp outdim[2] = {Nbas,Nbas};
	//std::cout << "NBAS: " << Nbas << endl;
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	int Nang = (1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	for (int i=0; i<Prm->SH_NRAD ; ++i)
	{
		for (int j=i; j<Prm->SH_NRAD ; ++j)
		{
			double S = GOverlap(Prm->RBFS[i*2],Prm->RBFS[j*2],Prm->RBFS[i*2+1],Prm->RBFS[j*2+1]);
			for (int l=0; l<Nang ; ++l)
			{
				int r = i*Nang + l;
				int c = j*Nang + l;
				SH_data[r*Nbas+c] = S;
				SH_data[c*Nbas+r] = S;
			}
		}
	}
	return SH;
}

static PyObject* Overlap_RBF(PyObject *self, PyObject  *args)
{
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &Prm_))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	int nbas = Prm->SH_NRAD;
	npy_intp outdim[2] = {nbas,nbas};
	PyObject* SRBF = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SRBF_data;
	SRBF_data = (double*) ((PyArrayObject*)SRBF)->data;
	for (int i=0; i<Prm->SH_NRAD ; ++i)
	{
		for (int j=0; j<Prm->SH_NRAD ; ++j)
		{
			SRBF_data[i*Prm->SH_NRAD+j] = GOverlap(Prm->RBFS[i*2],Prm->RBFS[j*2],Prm->RBFS[i*2+1],Prm->RBFS[j*2+1]);
		}
	}
	return SRBF;
}

static PyObject* Overlap_RBFS(PyObject *self, PyObject  *args)
{
	PyObject *Prm_, *RBF_obj;
	if (!PyArg_ParseTuple(args, "O!O!", &PyDict_Type, &Prm_, &PyArray_Type, &RBF_obj))
	return NULL;

	PyObject *RBF_array = PyArray_FROM_OTF(RBF_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	if (RBF_array == NULL) {
		Py_XDECREF(RBF_array);
		return NULL;
	}

	int Nele = (int)PyArray_DIM(RBF_array, 0);
	int BasMax = (int)PyArray_DIM(RBF_array, 1);
	int Bassz = (int)PyArray_DIM(RBF_array, 2);

	// std::cout << "Nele: " << Nele << " BasMax: " << BasMax << " L: " << L << std::endl;

	double *RBF = (double*)PyArray_DATA(RBF_array);
	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	int nbas = Prm->SH_NRAD;
	npy_intp outdim[3] = {Nele,nbas,nbas};
	PyObject* SRBF = PyArray_ZEROS(3, outdim, NPY_DOUBLE, 0);
	double *SRBF_data;
	SRBF_data = (double*) ((PyArrayObject*)SRBF)->data;
	// for (int i=0; i<100; ++i)
	// 	std::cout << RBF[i] << std::endl;
	for (int k=0; k<Nele ; ++k)
	{
		for (int i=0; i<Prm->SH_NRAD ; ++i)
		{
			for (int j=0; j<Prm->SH_NRAD ; ++j)
			{
				// std::cout << "k: " << k << " i: " << i << " RBFS_i: " << RBFS_data[k*12*2+i*2] << " " << RBFS_data[k*12*2+i*2+1] << " RBFS_j: " << RBFS_data[k*12*2+j*2] << " " << RBFS_data[k*12*2+j*2+1] << std::endl;
				SRBF_data[k*nbas*nbas+i*nbas+j] = GOverlap(RBF[k*BasMax*Bassz+i*2],RBF[k*BasMax*Bassz+j*2],RBF[k*BasMax*Bassz+i*2+1],RBF[k*BasMax*Bassz+j*2+1]);
			}
		}
	}
	Py_DECREF(RBF_array);
	return SRBF;
}

struct module_state {
	PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
	#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
	#define GETSTATE(m) (&_state)
	static struct module_state _state;
#endif

static PyObject * error_out(PyObject *m) {
		struct module_state *st = GETSTATE(m);
		PyErr_SetString(st->error, "something bad happened");
		return NULL;
}

static PyMethodDef EmbMethods[] =
{
	{"EmptyInterfacedFunction", EmptyInterfacedFunction, METH_VARARGS,
	"EmptyInterfacedFunction method"},
	{"Make_NListLinear", Make_NListLinear, METH_VARARGS,
	"Make_NListLinear method"},
	{"Make_NListNaive", Make_NListNaive, METH_VARARGS,
	"Make_NListNaive method"},
	{"Make_NLTensor", Make_NLTensor, METH_VARARGS,
	"Make_NLTensor method"},
	{"Make_TLTensor", Make_TLTensor, METH_VARARGS,
	"Make_TLTensor method"},
	{"DipoleAutoCorr", DipoleAutoCorr, METH_VARARGS,
	"DipoleAutoCorr method"},
	{"Make_DistMat", Make_DistMat, METH_VARARGS,
	"Make_DistMat method"},
	{"CountInRange", CountInRange, METH_VARARGS,
	"CountInRange method"},
	{"Make_DistMat_ForReal", Make_DistMat_ForReal, METH_VARARGS,
	"Make_DistMat_ForReal method"},
	{"Norm_Matrices", Norm_Matrices, METH_VARARGS,
	"Norm_Matrices method"},
	{"Make_SH", Make_SH, METH_VARARGS,
	"Make_SH method"},
	{"Make_SH_Transf", Make_SH_Transf, METH_VARARGS,
	"Make_SH_Transf method"},
	{"Make_Inv", Make_Inv, METH_VARARGS,
	"Make_Inv method"},
	{"Raster_SH", Raster_SH, METH_VARARGS,
	"Raster_SH method"},
	{"Overlap_SH", Overlap_SH, METH_VARARGS,
	"Overlap_SH method"},
	{"Overlap_RBF", Overlap_RBF, METH_VARARGS,
	"Overlap_RBF method"},
	{"Overlap_RBFS", Overlap_RBFS, METH_VARARGS,
	"Overlap_RBFS method"},
	{"Project_SH", Project_SH, METH_VARARGS,
	"Project_SH method"},
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

	static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
	    Py_VISIT(GETSTATE(m)->error);
	    return 0;
	}

	static int myextension_clear(PyObject *m) {
	    Py_CLEAR(GETSTATE(m)->error);
	    return 0;
	}

  static struct PyModuleDef moduledef = {
		PyModuleDef_HEAD_INIT,
		"MolEmb",     /* m_name */
		"A CAPI for TensorMol",  /* m_doc */
		sizeof(struct module_state),
		EmbMethods,    /* m_methods */
		NULL,                /* m_reload */
		myextension_traverse,                /* m_traverse */
		myextension_clear,                /* m_clear */
		NULL                /* m_free */
		};
	#pragma message("Compiling MolEmb for Python3x")
	#define INITERROR return NULL
	PyMODINIT_FUNC
	PyInit_MolEmb(void)
	{
		PyObject *m = PyModule_Create(&moduledef);
		if (m == NULL)
			INITERROR;
		struct module_state *st = GETSTATE(m);
		st->error = PyErr_NewException("MolEmb.Error", NULL, NULL);
		if (st->error == NULL) {
			Py_DECREF(m);
			INITERROR;
		}
		import_array();
		return m;
	}
#else
	PyMODINIT_FUNC
	initMolEmb(void)
	{
		(void) Py_InitModule("MolEmb", EmbMethods);
		/* IMPORTANT: this must be called */
		import_array();
		return;
	}
#endif
