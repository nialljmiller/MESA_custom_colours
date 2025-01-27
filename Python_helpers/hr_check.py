#!/usr/bin/env python3.8
####################################################
#
# Author: M Joyce
#
####################################################
import numpy as np
import glob
import matplotlib.pyplot as plt
import mesa_reader as mr

# Locate the history.data file
f = glob.glob('../LOGS/history.data')[0]

# Read the MESA data
md = mr.MesaData(f)

# Extract the necessary data for the plots
G = md.G
Gbp = md.Gbp
Grp = md.Grp
Teff = md.Teff
Log_L = md.log_L

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Gbp - Grp vs G in the first subplot
axes[0].plot(Gbp - Grp, G, 'go')
axes[0].set_xlabel('Gbp - Grp')
axes[0].set_ylabel('G')
axes[0].invert_yaxis()
axes[0].set_title('Color-Magnitude Diagram')

# Plot Teff vs Log_L in the second subplot
axes[1].plot(Teff, Log_L, 'go')
axes[1].set_xlabel('Teff (K)')
axes[1].set_ylabel('Log_L')
axes[1].invert_xaxis()
axes[1].set_title('Hertzsprung-Russell Diagram')

# Adjust layout and show the figure
plt.tight_layout()
plt.show()

