// Voxel class header file

#include <vector>

using namespace std;

class Voxel {
public:
	Voxel() {

	}
	~Voxel() {}
	void insert(double atom) {
		// Atoms.push_back(atom)
	}
	double pop(double atom) {
		// Search for atom, delete
	}
	vector<double> getAtoms() {
		return Atoms;
	}
private:
	vector<double> Atoms;
}
