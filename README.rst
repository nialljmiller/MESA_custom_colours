MESA Colors Project
====================

**MESA Colors** is an extension for the Modules for Experiments in Stellar Astrophysics (MESA) toolkit, designed to enable synthetic photometry calculations and provide additional stellar evolution diagnostics. By incorporating bolometric corrections and synthetic magnitudes, it allows for detailed analysis of stellar properties under various filter systems and atmospheric models.

## Features

- **Synthetic Magnitudes**: Compute magnitudes using arbitrary filter profiles.
- **Custom History Columns**: Append photometric diagnostics to the MESA history output.
- **SED Interpolation**: Interpolate between stellar models to generate spectral energy distributions (SEDs).
- **Flexible Filter System Integration**: Easily incorporate custom filter profiles for synthetic photometry.

---

## Installation

1. Clone or download the MESA Colors module to your MESA project directory.

2. Ensure your mesasdk path has been properly set

3. Clean and make module


```
./clean; ./mk
```


3. This will download a 35MB file with JWST and Gaia filter transmission curves as well as Kurucz 2003 stellar atmosphere models and some pre computed black body curves. These will be stored in /data

---
```
mesa_colors/
├── data/
│   ├── stellar_models/           # Stellar models (Kurucz, PHOENIX, etc.)
│   ├── filters/                  # Filter profiles (e.g., GAIA, JWST, etc.)
│   ├── extracted_marker          # Marker file to identify if zip file has been extracted ot /data properly

```
---


### Outputs

- **History Columns**: Bolometric magnitudes and fluxes will be appended to the MESA history output.
- **SED.csv**: If ``x_character_ctrl(4) = 'false'``. This will save csv files of each constructed SED.
 

---

## Directory Structure

```
mesa_colors/
├── src/
│   ├── run_star_extras.f90       # Main Fortran module
│   ├── utilities.f90             # Helper functions for interpolation and file handling
├── data/
│   ├── stellar_models/           # Stellar models (Kurucz, PHOENIX, etc.)
│   ├── filters/                  # Filter profiles (e.g., GAIA, HST, etc.)
├── inlist                        # Example inlist
```

---

## Dependencies

- MESA (Version 12778 or higher recommended)
- Fortran Compiler (e.g., gfortran, ifort)
- For "python_helpers"
   - Python3
      - numpy
      - matplotlib
      - mesa reader
---

## Acknowledgments

This project was inspired by the need for more detailed synthetic photometry in stellar evolution simulations. It incorporates insights from MESA's development community and builds upon the work of Rob Farmer and the MESA Team.


