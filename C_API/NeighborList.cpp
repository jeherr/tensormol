// Neighborlist

// Includes
#include <list>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

// Classes
class Atom {
public:
	Atom() {
		setID(-1);
		setN(0);
	}
	Atom(int ID, int N) {
		setID(ID);
		setN(N);
	}
	~Atom() {}

	void setID(int ID) {this->ID = ID;}
	void setN(int N) {this->N = N;}

	int getID(void) {return ID;}
	int getN(void) {return N;}

private:
	int ID; // Atom identification number
	int N;  // Atomic number
};

// Typedefs and definitions
typedef *(list<Atom>) cell_list_ptr;
typedef vector<vector<vector<cell_list_ptr>>> grid;

#define R_CUT 1; // Cutoff radius. 1 Angstrom for now

// Prototypes
int get_atomic_number(string);

// Main execution
int main() {
	// Read first two lines of xyz file
	int n_atoms;
	cin >> n_atoms;
	string comment;
	cin >> comment;

	// n_atoms x 4 vector that holds each atom's coordinates i=[0-2]
	// and the atomic number of the atom i=[3]
	vector<vector<double>> xyzs(natoms, {0,0,0,0});

	// xbuckets x ybuckets x zbuckets grid that holds the pointers
	// to the list of atoms that are inside of that bucket
	grid Heads;

	// Read in the atom coordinates
	double xmax, xmin, ymax, ymin, zmax, zmin;

	for (int i = 0; i < n_atoms; i++) {
		// Build xyzs vector, determine max/min xyz values
		string atom_name;
		cin >> atom_name >> xyzs[i][0] >> xyzs[i][1] >> xyzs[i][2];
		xyzs[3] = get_atomic_number(atom_name);
		if (i == 0 || xyzs[i][0] > xmax) xmax = xyzs[i][0];
		if (i == 0 || xyzs[i][1] > ymax) ymax = xyzs[i][1];
		if (i == 0 || xyzs[i][2] > zmax) zmax = xyzs[i][2];
		if (i == 0 || xyzs[i][0] < xmin) xmin = xyzs[i][0];
		if (i == 0 || xyzs[i][1] < ymin) ymin = xyzs[i][1];
		if (i == 0 || xyzs[i][2] < zmin) zmin = xyzs[i][2];
	}

	// Offset coordinates so that atoms are in quadrant I, close to the axes
	xmax += (-xmin + (R_CUT/2));
	ymax += (-ymin + (R_CUT/2));
	zmax += (-zmin + (R_CUT/2));

	for (int i = 0; i < n_atoms; i++) {
		xyzs[i][0] += (-xmin + (R_CUT/2));
		xyzs[i][1] += (-ymin + (R_CUT/2));
		xyzs[i][2] += (-zmin + (R_CUT/2));
	}

	// ==== Construct Heads grid ====
	// Number of buckets in each direction
	int xbuckets = (xmax / R_CUT) + 1;
	int ybuckets = (ymax / R_CUT) + 1;
	int zbuckets = (zmax / R_CUT) + 1;

	// Create a pointer to an empty list of atoms
	// for each cell in the Heads grid
	for (int z = 0; z < zbuckets; z++) {
		for (int y = 0; y < ybuckets; y++) {
			for (int x = 0; x < xbuckets; x++) {
				cell_list_ptr p;
				Heads[x][y][z] = p;
			}
		}
	}

	// ==== Construct all lists in grid ====
	for (int i = 0; i < n_atoms; i++) {
		Atom atom(i, xyzs[i][3]);
		int xtarget = xyzs[i][0] / R_CUT;
		int ytarget = xyzs[i][1] / R_CUT;
		int ztarget = xyzs[i][2] / R_CUT;

		Heads[xtarget][ytarget][ztarget]->push_back(atom);
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
