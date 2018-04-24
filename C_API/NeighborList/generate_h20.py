#!/usr/bin/env python2.7

import sys

'''
3
Water molecule
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
'''

if len(sys.argv) != 2:
	print "\nUsage:", sys.argv[0], "N_MOLECULES"
	print "\n\twhere N_MOLECULES corresponds to the number"
	print "\tof molecules along one axis of the three"
	print "\tdimensional grid of water molecules\n"
	exit(1)

obase = [0.00000, 0.00000, 0.11779]
h1base = [0.00000, 0.75545, -0.47116]
h2base = [0.00000, -0.75545, -0.47116]

oindex = [0.00000, 0.00000, 0.11779]
h1index = [0.00000, 0.75545, -0.47116]
h2index = [0.00000, -0.75545, -0.47116]

stride = 2.0
n_mols = int(sys.argv[1])

print (n_mols**3)*3
print "Box of water"

for i in range(n_mols):
	oindex[0] = obase[0] + stride*float(i)
	h1index[0] = h1base[0] + stride*float(i)
	h2index[0] = h2base[0] + stride*float(i)

	for j in range(n_mols):
		oindex[1] = obase[1] + stride*float(j)
		h1index[1] = h1base[1] + stride*float(j)
		h2index[1] = h2base[1] + stride*float(j)

		for k in range(n_mols):
			oindex[2] = obase[2] + stride*float(k)
			h1index[2] = h1base[2] + stride*float(k)
			h2index[2] = h2base[2] + stride*float(k)

			print "O", oindex[0], oindex[1], oindex[2]
			print "H", h1index[0], h1index[1], h1index[2]
			print "H", h2index[0], h2index[1], h2index[2]
