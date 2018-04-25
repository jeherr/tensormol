// Neighborlist

// Includes
//#include <Python.h>
#include <list>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <set>
#include <utility>
#include <cmath>

using namespace std;

// Classes
class Atom {
public:
	Atom() {
		setID(-1);
		setN(0);
		setXYZ(0,0,0);
	}
	Atom(int ID, int N, double x, double y, double z) {
		setID(ID);
		setN(N);
		setXYZ(x, y, z);
	}
	~Atom() {}

	void setID(int ID) {this->ID = ID;}
	void setN(int N) {this->N = N;}
	void setXYZ(double x, double y, double z) {
		XYZ.push_back(x);
		XYZ.push_back(y);
		XYZ.push_back(z);
	}

	int getID(void) {return ID;}
	int getN(void) {return N;}
	vector<double> getXYZ(void) {return XYZ;}
	double getX(void) {return XYZ[0];}
	double getY(void) {return XYZ[1];}
	double getZ(void) {return XYZ[2];}

private:
	int ID; // Atom identification number
	int N;  // Atomic number
	vector<double> XYZ;
};

// Typedefs and definitions
typedef list<Atom> *cell_list_ptr;
typedef vector<vector<vector<cell_list_ptr> > > grid;
typedef list<pair<int, int>> neighborlist;

#define R_CUT 15. // Cutoff radius (Angstroms)
#define pair pair<int,int>
#define Cell Heads[x][y][z]

// Prototypes
int get_atomic_number(string);
string get_atom_name(int);

// Main execution: Process input file and create the x by
// y by z grid, with each cell pointing to a list of Atoms
int main() {

	// Read first two lines of xyz file
	string n_atoms_str;
	getline(cin, n_atoms_str);
	int n_atoms = stoi(n_atoms_str);
	string comment;
	getline(cin, comment);

	// n_atoms x 4 vector that holds each atom's coordinates i=[0-2]
	// and the atomic number of the atom i=[3]
	vector<double> zerovector(4, 0.0);
	vector<vector<double> > xyzs(n_atoms, zerovector);

	// Read in the atom coordinates
	double xmax, xmin, ymax, ymin, zmax, zmin;

	// Build xyzs vector, determine max/min xyz values
	for (int i = 0; i < n_atoms; i++) {
		string atom_name;
		cin >> atom_name >> xyzs[i][0] >> xyzs[i][1] >> xyzs[i][2];
		xyzs[i][3] = get_atomic_number(atom_name);
		if (i == 0 || xyzs[i][0] > xmax) xmax = xyzs[i][0];
		if (i == 0 || xyzs[i][1] > ymax) ymax = xyzs[i][1];
		if (i == 0 || xyzs[i][2] > zmax) zmax = xyzs[i][2];
		if (i == 0 || xyzs[i][0] < xmin) xmin = xyzs[i][0];
		if (i == 0 || xyzs[i][1] < ymin) ymin = xyzs[i][1];
		if (i == 0 || xyzs[i][2] < zmin) zmin = xyzs[i][2];
	}

	neighborlist neighbors;

	for (int i = 0; i < n_atoms; i++) {
		for (int j = i + 1; j < n_atoms; j++) {
			double distance = sqrt( pow((xyzs[i][0] - xyzs[j][0]),2)
														+ pow((xyzs[i][1] - xyzs[j][1]),2)
														+ pow((xyzs[i][2] - xyzs[j][2]),2) );
			if (distance < R_CUT && distance != 0.0) {
				neighbors.push_back(make_pair(i, j));
			}
		}
	}

	for (auto it = neighbors.begin(); it != neighbors.end(); it++) {
		printf("%s: %9.6f %9.6f %9.6f ||| %s: %9.6f %9.6f %9.6f ... %9.6f\n", get_atom_name(xyzs[it->first][3]).c_str(), xyzs[it->first][0],
					xyzs[it->first][1], xyzs[it->first][2],
					get_atom_name(xyzs[it->second][3]).c_str(), xyzs[it->second][0],
					xyzs[it->second][1], xyzs[it->second][2],
					sqrt( pow((xyzs[it->first][0] - xyzs[it->second][0]), 2)
					+ pow((xyzs[it->first][1] - xyzs[it->second][1]), 2)
					+ pow((xyzs[it->first][2] - xyzs[it->second][2]), 2) )
					);
	}

}

int get_atomic_number(string s) {
	unordered_map<string, int> Atoms = {
		{"X", 0},
		{"H", 1},
		{"He", 2},
		{"Li", 3},
		{"Be", 4},
		{"B", 5},
		{"C", 6},
		{"N", 7},
		{"O", 8},
		{"F", 9}
	};
	if (Atoms.find(s) != Atoms.end())
		return Atoms[s];
	else
		return Atoms["X"];
}

string get_atom_name(int x) {
	unordered_map<int, string> Atoms = {
		{0, "X"},
		{1, "H"},
		{2, "He"},
		{3, "Li"},
		{4, "Be"},
		{5, "B"},
		{6, "C"},
		{7, "N"},
		{8, "O"},
		{9, "F"}
	};
	if (Atoms.find(x) != Atoms.end())
		return Atoms[x];
	else
		return Atoms[0];
}
