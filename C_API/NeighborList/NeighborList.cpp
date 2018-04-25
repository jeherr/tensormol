// Neighborlist

// Includes
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
	/*for (int x = 0; x < Heads.size(); x++) {
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
	}*/

	neighborlist neighbors = compute_neighbor_list(Heads);

/*
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
*/
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
				vector<vector<int>> all_cells;
				set<vector<int>> to_visit;

				// Cell * (not included in to_visit, dealt with afterwards)
				vector<int> current_coords = {x, y, z};

				// Cell C (0)
				vector<int> c_coords = {x + 1, y - 1, z - 1};
				all_cells.push_back(c_coords);

				// Cell F (1)
				vector<int> f_coords = {x + 1, y, z - 1};
				all_cells.push_back(f_coords);

				// Cell I (2)
				vector<int> i_coords = {x + 1, y + 1, z - 1};
				all_cells.push_back(i_coords);

				// Cell K (3)
				vector<int> k_coords = {x, y - 1, z};
				all_cells.push_back(k_coords);

				// Cell L (4)
				vector<int> l_coords = {x + 1, y - 1, z};
				all_cells.push_back(l_coords);

				// Cell N (5)
				vector<int> n_coords = {x + 1, y, z};
				all_cells.push_back(n_coords);

				// Cell Q (6)
				vector<int> q_coords = {x + 1, y + 1, z};
				all_cells.push_back(q_coords);

				// Cell S (7)
				vector<int> s_coords = {x, y - 1, z + 1};
				all_cells.push_back(s_coords);

				// Cell T (8)
				vector<int> t_coords = {x + 1, y - 1, z + 1};
				all_cells.push_back(t_coords);

				// Cell V (9)
				vector<int> v_coords = {x, y, z + 1};
				all_cells.push_back(v_coords);

				// Cell W (10)
				vector<int> w_coords = {x + 1, y, z + 1};
				all_cells.push_back(w_coords);

				// Cell Y (11)
				vector<int> y_coords = {x, y + 1, z + 1};
				all_cells.push_back(y_coords);

				// Cell Z (12)
				vector<int> z_coords = {x + 1, y + 1, z + 1};
				all_cells.push_back(z_coords);

				/* Inserting into the to_visit */
				for (size_t i = 0; i < all_cells.size(); i++) {
					if (all_cells[i][0] >= 0 && all_cells[i][0] < xbuckets &&
							all_cells[i][1] >= 0 && all_cells[i][1] < ybuckets &&
							all_cells[i][2] >= 0 && all_cells[i][2] < zbuckets) {
						to_visit.insert(all_cells[i]);
					}
				}

				for (auto it = Cell->begin(); it != Cell->end(); it++) {
					//cout << x << " " << y << " " << z << " " << "\n";
					for (auto jt = to_visit.begin(); jt != to_visit.end(); jt++) {
						int x2 = (*jt)[0]; int y2 = (*jt)[1]; int z2 = (*jt)[2];
						//cout << x2 << " " << y2 << " " << z2 << " | ";
						for (auto kt = Heads[x2][y2][z2]->begin(); kt != Heads[x2][y2][z2]->end(); kt++) {
							double distance = sqrt( pow((it->getX() - kt->getX()), 2) + pow((it->getY() - kt->getY()), 2) + pow((it->getZ() - kt->getZ()), 2) );
							if (distance < R_CUT && distance != 0.0) {
								neighbors.push_back(make_pair(it->getID(), kt->getID()));
							}
						}
					}
					//cout << "\n";
				}
				for (auto it = Cell->begin(); it != Cell->end(); it++) {
					auto curr = it;
					curr++;
					for (auto jt = curr; jt != Cell->end(); jt++) {
						double distance = sqrt( pow((it->getX() - jt->getX()), 2) + pow((it->getY() - jt->getY()), 2) + pow((it->getZ() - jt->getZ()), 2) );
						if (distance < R_CUT && distance != 0.0) {
							neighbors.push_back(make_pair(it->getID(), jt->getID()));
						}
					}
				}
			}
		}
	}

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
