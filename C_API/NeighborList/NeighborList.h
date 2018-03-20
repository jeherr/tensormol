// NeighborList header file

#include "Voxel.h"

using namespace std;

class NeighborList {
public:
	NeighborList() {}
	~NeighborList() {}

private:
	vector<vector<vector<Voxel> > > VoxelLattice;
}
