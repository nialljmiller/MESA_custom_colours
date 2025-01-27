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
# from matplotlib.patches import Rectangle
# from matplotlib.collections import PatchCollection
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# import matplotlib.patches as patches

# import argparse

#sys.path.append('../py_mesa_reader/')
import mesa_reader as mr

f = glob.glob('../LOGS/history.data')[0]

#print('this is f: ', f)

md = mr.MesaData(f)

Teff = md.Teff
Log_L = md.log_L

plt.plot(Teff, Log_L, 'go')
plt.xlabel('Teff (K)')
plt.ylabel('Log_L')
plt.gca().invert_xaxis()
plt.show()
plt.close()
