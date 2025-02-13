MESA Colors Project
====================

**MESA Colors** is an extension for the Modules for Experiments in Stellar Astrophysics (MESA) toolkit, designed to enable synthetic photometry calculations and provide additional stellar evolution diagnostics. By incorporating bolometric corrections and synthetic magnitudes, it allows for detailed analysis of stellar properties under various filter systems and atmospheric models.

Features
--------

- **Synthetic Magnitudes**: Compute magnitudes using arbitrary filter profiles.
- **Custom History Columns**: Append photometric diagnostics to the MESA history output.
- **SED Interpolation**: Interpolate between stellar models to generate spectral energy distributions (SEDs).
- **Flexible Filter System Integration**: Easily incorporate custom filter profiles for synthetic photometry.

Installation
------------

1. Clone or download the MESA Colors module to your MESA project directory.
2. Ensure your `mesasdk` path is properly set.
3. Clean and compile the module:

   .. code-block:: bash
      
      ./clean; ./mk

4. The script will download a 35MB file containing JWST and Gaia filter transmission curves, Kurucz 2003 stellar atmosphere models, and precomputed blackbody curves. These files will be stored in the `data/` directory.

Directory Structure
-------------------

.. code-block:: text

   mesa_colors/
   ├── src/
   │   ├── run_star_extras.f90       # Main Fortran module
   │   ├── utilities.f90             # Helper functions for interpolation and file handling
   ├── data/
   │   ├── stellar_models/           # Stellar models (Kurucz, PHOENIX, etc.)
   │   ├── filters/                  # Filter profiles (e.g., GAIA, HST, JWST, etc.)
   │   ├── extracted_marker          # Marker file to confirm data extraction
   ├── inlist                        # Example inlist

Inlist Options
-------

- **x_character_ctrl(1)** : 'data/stellar_models/Kurucz2003all/'   -- Stellar atmosphere model - http://svo2.cab.inta-csic.es/theory/newov2/
- **x_character_ctrl(2)** : 'data/filters/GAIA/GAIA'               -- Photometric filter system - http://svo2.cab.inta-csic.es/theory/fps/
- **x_character_ctrl(3)** : 'data/stellar_models/vega_flam.csv'    -- Vega SED for Vega photometric system
- **x_character_ctrl(4)** : 'false'                                -- Save csv files of each constructed SED?


Outputs
-------

- **History Columns**: Bolometric magnitudes and fluxes will be appended to the MESA history output.
- **SED.csv**: If ``x_character_ctrl(4) = 'false'``, this will save CSV files of each constructed SED.

Dependencies
------------

- **MESA** (Version 12778 or higher recommended)
- **Fortran Compiler** (e.g., `gfortran`, `ifort`)
- **Python Helpers**:
  - Python 3
  - `numpy`
  - `matplotlib`
  - `mesa_reader`

Acknowledgments
---------------

This project was inspired by the need for more detailed synthetic photometry in stellar evolution simulations. It incorporates insights from MESA's development community and builds upon the work of Rob Farmer and the MESA Team.