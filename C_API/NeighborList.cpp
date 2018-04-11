// Neighborlist

// Includes
#include <list>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_set>

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

#define R_CUT 1.0 // Cutoff radius. 1 Angstrom for now
#define pair pair<int,int>
#define Cell Heads[x][y][z]

// Prototypes
int get_atomic_number(string);
neighborlist compute_neighbor_list(grid);

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

	// xbuckets x ybuckets x zbuckets grid that holds the pointers
	// to the list of atoms that are inside of that bucket
	list<Atom> p;
	cell_list_ptr placeholder = &p;
	vector<cell_list_ptr> vec1(zbuckets, placeholder);
	vector<vector<cell_list_ptr>> vec2(ybuckets, vec1);
	grid Heads(xbuckets, vec2);

	vector<list<Atom>> lists;
	for (int i = 0; i < (xbuckets+1)*(ybuckets+1)*(zbuckets+1); i++) {
		list<Atom> l;
		lists.push_back(l);
	}

	// Putting unique cell_list_ptr's into each cell
	int count = 0;
	for (int x = 0; x < xbuckets; x++) {
		for (int y = 0; y < ybuckets; y++) {
			for (int z = 0; z < zbuckets; z++) {
				Cell = &lists[count++];
			}
		}
	}

	// ==== Construct all lists in grid ====
	for (int i = 0; i < n_atoms; i++) {
		Atom atom(i, xyzs[i][3], xyzs[i][0], xyzs[i][1], xyzs[i][2]);
		int x = xyzs[i][0] / R_CUT;
		int y = xyzs[i][1] / R_CUT;
		int z = xyzs[i][2] / R_CUT;
		Cell->push_back(atom);
	}
	/* ==== Testing ==== */
	for (int x = 0; x < Heads.size(); x++) {
		for (int y = 0; y < Heads[x].size(); y++) {
			for (int z = 0; z < Heads[x][y].size(); z++) {
				cout << x << ' ' << y << ' ' << z << ": ";
				for (auto it = Cell->begin(); it != Cell->end(); it++) {
					cout << it->getID() << ' ';
				}
				cout << '\n';
			}
			cout << '\n';
		}
		cout << "\n\n\n";
	}

	neighborlist neighbors = compute_neighbor_list(Heads);

}

neighborlist compute_neighbor_list(grid Heads) {
	int xbuckets = (int)Heads.size();
	int ybuckets = (int)Heads[0].size();
	int zbuckets = (int)Heads[0][0].size();

	neighborlist neighbors;

	/* =====================
	Cell Naming Conventions:
	* is current cell
	Cells denoted by =.= are checked if they exist
	Cells denoted by _._ are skipped
	 ___ ___ ___    ___ ___ ___    ___ ___ ___
	|_G_|_H_|=I=|  |_O_|_P_|=Q=|  |_X_|=Y=|=Z=|
	|_D_|_E_|=F=|  |_M_|=*=|=N=|  |_U_|=V=|=W=|
	|_A_|_B_|=C=|  |_J_|=K=|=L=|  |_R_|=S=|=T=|
	Bottom Layer   Middle Layer   Top Layer (z -1, +0, +1)

	Y -1, +0, +1
	^
	|
	+-----> X -1, +0, +1

	===================== */

	for (int x = 0; x < xbuckets; x++) {
		for (int y = 0; y < ybuckets; y++) {
			for (int z = 0; z < zbuckets; z++) {
				unordered_set<vector<int>> to_visit;

				// Cell * (always included in to_visit)
				vector<int> current_coords {x, y, z};
				to_visit.insert(current_coords);

				// Cell C
				vector<int> c_coords {x + 1, y - 1, z - 1};
				/* TODO NEXT ==> ALL THE CONDITIONS FOR WHETHER TO INSERT
				THE COORDS INTO THE TO_VISIT VECTOR */

				// Cell F
				vector<int> f_coords {x + 1, y, z - 1};

				// Cell I
				vector<int> i_coords {x + 1, y + 1, z - 1};

				// Cell K
				vector<int> k_coords {x, y - 1, z};

				// Cell L
				vector<int> l_coords {x + 1, y - 1, z};

				// Cell N
				vector<int> n_coords {x + 1, y, z};

				// Cell Q
				vector<int> q_coords {x + 1, y + 1, z};

				// Cell S

				// Cell T

				// Cell V

				// Cell W

				// Cell Y

				// Cell Z
				

				for (auto it = Cell->begin(); it != Cell->end(); it++) {

				}
			}
		}
	}


	/* ==== neighbors.insert(make_pair(atom_id_1, atom_id_2)); ==== */

	return neighbors;
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
