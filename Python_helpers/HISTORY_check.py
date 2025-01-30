#!/usr/bin/env python3.8
####################################################
#
# Author: M Joyce, Modified by N. Miller
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

# Extract relevant data
G = md.G
Gbp = md.Gbp
Grp = md.Grp
Teff = md.Teff
Log_L = md.log_L
Log_g = md.log_g
Log_R = md.log_R
Star_Age = md.star_age
Mag_bol = md.Mag_bol
Flux_bol = np.log10(md.Flux_bol)  # Convert to log scale

# Create a figure with multiple subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot Gbp - Grp vs G (CMD)
axes[0, 0].plot(Gbp - Grp, G, 'go')
axes[0, 0].set_xlabel('Gbp - Grp')
axes[0, 0].set_ylabel('G')
axes[0, 0].invert_yaxis()
#axes[0, 0].set_title('Color-Magnitude Diagram')

# Plot Log Age vs Log Flux_bol (Flux Bolometric vs Age)
axes[0, 1].plot(Gbp - Grp, G - Grp, 'go')
axes[0, 1].set_xlabel('Gbp - Grp')
axes[0, 1].set_ylabel('G - Grp')
#axes[2, 1].set_title('Flux Bolometric vs Age')


# Plot Log Age vs Log Flux_bol (Flux Bolometric vs Age)
axes[0, 2].plot(Gbp - Grp, G - Gbp, 'go')
axes[0, 2].set_xlabel('Gbp - Grp')
axes[0, 2].set_ylabel('G - Gbp')
#axes[2, 1].set_title('Flux Bolometric vs Age')

# Plot Teff vs Log_L (HRD)
axes[1, 0].plot(Teff, Log_L, 'go')
axes[1, 0].set_xlabel('Teff (K)')
axes[1, 0].set_ylabel('Log_L')
axes[1, 0].invert_xaxis()
#axes[0, 1].set_title('Hertzsprung-Russell Diagram')


# Plot Teff vs Log_R (Radius vs Temperature)
axes[1, 1].plot(Teff, Log_R, 'go')
axes[1, 1].set_xlabel('Teff (K)')
axes[1, 1].set_ylabel('Log_R')
axes[1, 1].invert_xaxis()
#axes[1, 1].set_title('Radius vs Temperature')

# Plot Teff vs Log_g (Surface Gravity vs Temperature)
axes[1, 2].plot(Teff, Log_g, 'go')
axes[1, 2].set_xlabel('Teff (K)')
axes[1, 2].set_ylabel('Log_g')
axes[1, 2].invert_xaxis()
#axes[0, 2].set_title('Surface Gravity vs Temperature')

# Plot Log Age vs Log Flux_bol (Flux Bolometric vs Age)
axes[2, 0].plot(np.log10(Star_Age), Flux_bol, 'go')
axes[2, 0].set_xlabel('Log Age (yr)')
axes[2, 0].set_ylabel('Log Flux_bol')
#axes[2, 0].set_title('Flux Bolometric vs Age')

# Plot Log Age vs Mag_bol (Bolometric Magnitude vs Time)
axes[2, 1].plot(np.log10(Star_Age), Mag_bol, 'go')
axes[2, 1].set_xlabel('Log Age (yr)')
axes[2, 1].set_ylabel('Mag_bol')
axes[2, 1].invert_yaxis()
#axes[1, 0].set_title('Bolometric Magnitude vs Age')


# Plot Log Age vs G (Gaia G-band vs Age)
axes[2, 2].plot(np.log10(Star_Age), G, 'go')
axes[2, 2].set_xlabel('Log Age (yr)')
axes[2, 2].set_ylabel('G')
axes[2, 2].invert_yaxis()
#axes[1, 2].set_title('Gaia G-band vs Age')




# Adjust layout and show the figure
#plt.tight_layout()
plt.show()
