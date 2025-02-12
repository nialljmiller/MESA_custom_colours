MESA Colors Project
====================

**MESA Colors** is an extension for the Modules for Experiments in Stellar Astrophysics (MESA) toolkit, designed to enable synthetic photometry calculations and provide additional stellar evolution diagnostics. By incorporating bolometric corrections and synthetic magnitudes, it allows for detailed analysis of stellar properties under various filter systems and atmospheric models.

## Features

- **Bolometric Corrections**: Calculate bolometric magnitudes and fluxes from stellar models.
- **Synthetic Magnitudes**: Compute magnitudes using arbitrary filter profiles.
- **Custom History Columns**: Append photometric diagnostics to the MESA history output.
- **SED Interpolation**: Interpolate between stellar models to generate spectral energy distributions (SEDs).
- **Flexible Filter System Integration**: Easily incorporate custom filter profiles for synthetic photometry.

---

## Installation

1. Clone or download the MESA Colors module to your MESA project directory.

   ```bash
   git clone <repository_url> mesa_colors
   ```

2. Add `run_star_extras` to your project’s Fortran modules. Ensure it is linked with MESA's internal libraries (e.g., `star_lib`, `math_lib`).

3. Copy the example `inlist` to your MESA project folder and modify it as needed.

4. Make sure the necessary data files (e.g., stellar models, filter profiles) are in the specified directories (`data/stellar_models/` and `data/filters/`).

---

## Quick Start

### 1. Setting Up the Inlist

Use the provided example `inlist` as a template. Key parameters include:

- **Stellar model directory**: Specify the path to stellar models in `x_character_ctrl(1)`.
- **Filter profiles**: Define the path to filter profiles in `x_character_ctrl(2)`. 
NOTE: THIS SHOULD BE THE INSTRUMENT PATH i.e. data/filters/JWST/**MIRI**

```fortran
&controls
   initial_mass = 19.0d0
   initial_z = 0.014d0
   x_character_ctrl(1) = 'data/stellar_models/Kurucz2003all/'
   x_character_ctrl(2) = 'data/filters/GAIA/GAIA'
/
```

### 2. Running the Simulation

Run MESA as usual. The module integrates seamlessly into the MESA workflow:

```bash
./clean ./make ./rn
```

### 3. Outputs

- **History Columns**: Bolometric magnitudes and fluxes will be appended to the MESA history output.
- **Synthetic Magnitudes**: Computed for all filters defined in `custom_colors_history_columns.list`.

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

---

## Key Subroutines and Functions

### **Bolometric Magnitudes**

- **`CalculateBolometricFlux`**: Computes the integrated bolometric flux and magnitude using trapezoidal integration.

- **`CalculateSyntheticMagnitude`**: Convolves SEDs with filter profiles to compute synthetic magnitudes.

### **SED Interpolation**

- **`InterpolateSED`**: Interpolates between stellar models to produce continuous SEDs for arbitrary physical parameters.

- **`LoadLookupTable`**: Reads the stellar model lookup table to find the closest matching models.

### **File Utilities**

- **`LoadFilter`**: Loads custom filter profiles for synthetic photometry.

- **`LoadSED`**: Loads SED data for stellar models.

---

## Example Workflow

1. Prepare the `stellar_models/` and `filters/` directories with your data.
2. Define desired filters in `custom_colors_history_columns.list`.
3. Run the MESA simulation.
4. Analyze the outputs in the MESA history files.

---

## Contributing

Contributions are welcome! Please follow the existing coding style and submit a pull request. For major changes, open an issue to discuss the proposed changes.

---

## License

This project is licensed under the GNU General Public License (GPL) v2 or later. See the `LICENSE` file for details.

---

## Acknowledgments

This project was inspired by the need for more detailed synthetic photometry in stellar evolution simulations. It incorporates insights from MESA's development community and builds upon the work of Rob Farmer and the MESA Team.


