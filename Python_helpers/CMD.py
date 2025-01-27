#!/usr/bin/env python3.8
####################################################
#
# Author: M Joyce
#
####################################################
import numpy as np
import glob
import sys
import subprocess
import matplotlib.pyplot as plt
import mesa_reader as mr

# Mag_bol
# Flux_bol
# Gbp_bright
# Gbp
# Gbp_faint
# G
# Grp                                     
# Grvs 



f = glob.glob('../LOGS/history.data')[0]

#print('this is f: ', f)

md = mr.MesaData(f)

G = md.G
Gbp = md.Gbp
Grp =md.Grp

plt.plot(Gbp-Grp, G, 'go')
plt.xlabel('Gbp - Grp')
plt.ylabel('G')
plt.gca().invert_yaxis()
plt.show()
plt.close()
