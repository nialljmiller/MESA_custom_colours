import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import matplotlib.pyplot as plt
import argparse

# Physical constants (cgs units)
h = 6.62607015e-27   # Planck constant (erg·s)
c = 2.99792458e10    # Speed of light (cm/s)
k_B = 1.380649e-16   # Boltzmann constant (erg/K)
def read_mesa_history(filename):
    """
    Reads the MESA history file, neatly prints the content,
    and parses it into a pandas DataFrame.
    """
    print(f"Reading MESA history file: {filename}")
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: History file '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading history file: {e}")
        sys.exit(1)

    # Print the contents of the history file
    print("\n--- History File Contents (Preview) ---")
    for idx, line in enumerate(lines[:20]):  # Print the first 20 lines for brevity
        # Truncate long lines and print with line numbers
        line_content = line.strip()
        truncated_line = (line_content[:120] + '...') if len(line_content) > 120 else line_content
        print(f"{idx + 1:>4}: {truncated_line}")
    print("--- End of History File Preview ---\n")

    # Find the header line
    header_line_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('model_number'):
            header_line_idx = idx
            break
    if header_line_idx is None:
        print("Error: Header line with column names not found in history file.")
        sys.exit(1)

    print(f"Header found at line {header_line_idx + 1}:")
    print(lines[header_line_idx].strip())  # Print header without truncation


    headers = lines[header_line_idx].strip().split()

    # Parse the file into a DataFrame
    try:
        data = pd.read_csv(
            filename,
            skiprows=header_line_idx + 1,
            sep=r'\s+',
            names=headers,
            comment='#',
            engine='python',
            )#nrows=1)            
    except Exception as e:
        print(f"Error parsing history file: {e}")
        sys.exit(1)

    print(f"History file successfully parsed with {len(data)} entries.")
    return data

def get_closest_stellar_models(teff, log_g, lookup_table):
    print(f"Finding closest stellar models for Teff={teff}, log_g={log_g}")
    distances = np.sqrt((lookup_table[' teff'] - teff)**2 + (lookup_table[' logg'] - log_g)**2)
    closest_indices = np.argsort(distances)[:2]
    if len(closest_indices) < 2:
        raise ValueError("Not enough models for interpolation.")
    print(f"Closest models found: {closest_indices}")
    return lookup_table.iloc[closest_indices]

def interpolate_sed(teff, log_g, stellar_model_dir, lookup_table):
    print(f"Interpolating SED for Teff={teff}, log_g={log_g}")
    closest_models = get_closest_stellar_models(teff, log_g, lookup_table)

    print(f"Closest models for interpolation:\n {closest_models}")

    # Check for a perfect match
    if (closest_models.iloc[0][' teff'] == teff) and (closest_models.iloc[0][' logg'] == log_g):
        print("Perfect match found for the stellar model.")
        sed_path = os.path.join(stellar_model_dir, closest_models.iloc[0]['#file_name'])
        sed = pd.read_csv(sed_path, sep='\s+', comment='#', header=None, names=['wavelength', 'flux'])
        return sed['wavelength'], sed['flux']

    sed1_path = os.path.join(stellar_model_dir, closest_models.iloc[0]['#file_name'])
    sed2_path = os.path.join(stellar_model_dir, closest_models.iloc[1]['#file_name'])
    print(f"Loading SEDs: {sed1_path}, {sed2_path}")

    sed1 = pd.read_csv(sed1_path, sep='\s+', comment='#', header=None, names=['wavelength', 'flux'])
    sed2 = pd.read_csv(sed2_path, sep='\s+', comment='#', header=None, names=['wavelength', 'flux'])

    print(f"SED1 head:\n{sed1.head()}")
    print(f"SED2 head:\n{sed2.head()}")

    # Check if the two closest models are identical in Teff
    if closest_models.iloc[0][' teff'] == closest_models.iloc[1][' teff']:
        print("Closest models have identical Teff; skipping interpolation.")
        return sed1['wavelength'], sed1['flux']  # Return one model directly

    # Interpolation weights
    weight1 = np.abs(closest_models.iloc[1][' teff'] - teff) / (np.abs(closest_models.iloc[1][' teff'] - closest_models.iloc[0][' teff']))
    weight2 = 1.0 - weight1
    print(f"Interpolation weights: weight1={weight1}, weight2={weight2}")

    # Interpolate the flux
    flux = weight1 * sed1['flux'] + weight2 * sed2['flux']
    return sed1['wavelength'], flux


def calculate_bolometric_magnitude(wavelength, flux):
    print("Calculating bolometric magnitude...")
    bolometric_flux = np.trapz(flux, wavelength)
    bolometric_magnitude = -2.5 * np.log10(bolometric_flux) - 48.6
    print(f"Bolometric magnitude calculated: {bolometric_magnitude}")
    return bolometric_magnitude
    
def convolve_with_filter(wavelength, flux, filter_dir, vega_sed):
    print(f"Convolving SED with filters in: {filter_dir}")
    filter_files = [f for f in os.listdir(filter_dir) if f.endswith('.dat')]
    print(f"Found {len(filter_files)} filter files: {filter_files}")

    # Convert wavelength to cm and flux to erg/s/cm²/Hz
    wavelength_cm = wavelength * 1e-8
    flux_nu = flux * (wavelength_cm**2) / c
    results = []
    summary = []

    vega_interp = interp1d(vega_sed['wavelength'], vega_sed['flux'], bounds_error=False, fill_value=0)

    for filter_file in filter_files:
        filter_path = os.path.join(filter_dir, filter_file)
        filter_data = pd.read_csv(filter_path, sep=',', header=0, names=['wavelength', 'transmission'])
        
        # Interpolate Vega SED onto the filter wavelength grid
        vega_flux_at_filter = vega_interp(filter_data['wavelength'])
        # Calculate λ_eff using the Vega spectrum
        numerator = np.trapz(filter_data['wavelength'] * filter_data['transmission'] * vega_flux_at_filter, filter_data['wavelength'])
        denominator = np.trapz(filter_data['transmission'] * vega_flux_at_filter, filter_data['wavelength'])
        lambda_eff = numerator / denominator

        peak_wavelength = filter_data['wavelength'][filter_data['transmission'].idxmax()]
        filter_interp = interp1d(filter_data['wavelength'], filter_data['transmission'], bounds_error=False, fill_value=0)
        filter_response = filter_interp(wavelength)

        # Restrict integration to valid filter response range
        valid = filter_response > 0
        numerator = np.trapz(flux_nu[valid] * filter_response[valid], wavelength[valid])
        denominator = np.trapz(filter_response[valid], wavelength[valid])

        if denominator == 0:
            m_AB = np.nan
            f_lambda = np.nan
            warning = f"Warning: Filter {filter_file} has no valid transmission in the SED range."
        else:
            f_nu = numerator / denominator
            m_AB = -2.5 * np.log10(f_nu) - 48.6

            # Calculate f_lambda at the peak wavelength
            f_lambda = f_nu * c / (peak_wavelength * 1e-8)**2  # Convert peak_wavelength to cm
            warning = None
        
        filter_name = filter_file.split('.')[0]
        results.append((m_AB, f_nu, f_lambda, lambda_eff, peak_wavelength, filter_name))
        summary.append({
            "Filter": filter_name,
            "λ_eff": lambda_eff,
            "Peak Wavelength": peak_wavelength,
            "Magnitude (m_AB)": m_AB,
            "Flux (f_nu)": f_nu if denominator != 0 else None,
            "Flux (f_lambda)": f_lambda if denominator != 0 else None,
            "Warning": warning
        })
    
        def format_value(val):
            """Formats values with up to 10 decimal places; switches to scientific if below 1e-10."""
            if val is None or np.isnan(val):
                return "N/A"
            elif abs(val) < 1e-10:
                return f"{val:.3e}"  # Use scientific notation
            else:
                return f"{val:.10f}".rstrip('0').rstrip('.')  # Remove trailing zeros

    print("\nSummary of Results:")
    print("-" * 50)
    for s in summary:
        print(f"Filter: {s['Filter']}")
        print(f"  Effective Wavelength: {format_value(s['λ_eff'])}")
        print(f"  Peak Wavelength: {format_value(s['Peak Wavelength'])}")
        print(f"  Magnitude (m_AB): {format_value(s['Magnitude (m_AB)'])}")
        print(f"  Flux (f_nu): {format_value(s['Flux (f_nu)'])}")
        print(f"  Flux (f_lambda): {format_value(s['Flux (f_lambda)'])}")
        if s['Warning']:
            print(f"  {s['Warning']}")
        print("-" * 50)

    return results


def plot_magnitudes(all_filter_results, output_file):
    """
    Plots magnitudes against their corresponding peak wavelengths.

    Parameters:
        all_filter_results (list): List of lists containing (magnitude, peak_wavelength) tuples.
        output_file (str): Path to the output file used to save the history DataFrame.
    """
    print("Plotting magnitudes...")
    plt.figure(figsize=(10, 10))

    for model_idx, filter_results in enumerate(all_filter_results):
        magnitudes, wavelengths = zip(*filter_results)
        plt.scatter(wavelengths, magnitudes, label=f"Model {model_idx + 1}", alpha=0.6)

    plt.xlabel("Wavelength (Angstroms)")
    plt.ylabel("AB Magnitude")
    plt.gca().invert_yaxis()  # Magnitudes are inverted (smaller is brighter)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plot_file = output_file.replace('.csv', '_magnitudes_plot.png')
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.show()

def load_and_resolve_paths(filepath):
    with open(filepath, 'r') as file:
        paths = {
            key.strip(): value.strip()
            for line in file if line.strip() and not line.startswith('#')
            for key, value in [line.strip().split('=', 1)]
        }
    base_dir = os.path.expanduser(paths['base_dir'])
    return {key: os.path.join(base_dir, value) if key != 'base_dir' else base_dir
            for key, value in paths.items()}


def main():
    resolved_paths = load_and_resolve_paths('dir_inlist.txt')
    
    # Extract variables from resolved_paths
    base_dir = resolved_paths['base_dir']
    history_file = resolved_paths['history_file']
    stellar_model = resolved_paths['stellar_model']
    instrument = resolved_paths['instrument']
    vega_sed_file = resolved_paths['vega_sed_file']
    
    vega_sed = pd.read_csv(vega_sed_file, sep=',', names=['wavelength', 'flux'], comment='#')#https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas
    print(f"Starting main process...")
    lookup_table_file = os.path.join(stellar_model, 'lookup_table.csv')
    lookup_table = pd.read_csv(lookup_table_file)
    print(f"Lookup table loaded: {lookup_table_file}")

    bolometric_magnitudes = []
    all_filter_results = []

    history = read_mesa_history(history_file)
    print(f"Processing {len(history)} models from history file...")

    for idx, row in history.iterrows():
        print(f"Processing model {idx + 1}/{len(history)}")
        teff = row.get('Teff')
        log_g = row.get('log_g')
        log_R = row.get('log_R')        
        metallicity = row.get('initial_feh', 0.0)

        wavelength, flux = interpolate_sed(teff, log_g, stellar_model, lookup_table)
        bol_mag = calculate_bolometric_magnitude(wavelength, flux)
        bolometric_magnitudes.append(bol_mag)

        filter_results = convolve_with_filter(wavelength, flux, instrument, vega_sed)
        all_filter_results.append(filter_results)  # Store tuples of (mag, peak_wavelength)

    # Add bolometric magnitudes to history
    history['bolometric_magnitude'] = bolometric_magnitudes
    # Add filter results as separate columns in the DataFrame
    for idx, filter_results in enumerate(all_filter_results):
        for mag, flux_nu, flux_lambda, lambda_eff, peak_wavelength, filter_name in filter_results:
            # Column for magnitude
            mag_col_name = f"mag_{filter_name}"
            if mag_col_name not in history:
                history[mag_col_name] = np.nan
            history.at[idx, mag_col_name] = mag

            # Column for flux_nu
            flux_nu_col_name = f"flux_nu_{filter_name}"
            if flux_nu_col_name not in history:
                history[flux_nu_col_name] = np.nan
            history.at[idx, flux_nu_col_name] = flux_nu

            # Column for flux_lambda
            flux_lambda_col_name = f"flux_lambda_{filter_name}"
            if flux_lambda_col_name not in history:
                history[flux_lambda_col_name] = np.nan
            history.at[idx, flux_lambda_col_name] = flux_lambda

            # Column for lambda_eff
            lambda_eff_col_name = f"lambda_eff_{filter_name}"
            if lambda_eff_col_name not in history:
                history[lambda_eff_col_name] = np.nan
            history.at[idx, lambda_eff_col_name] = lambda_eff

    # Save results
    output_file = base_dir + 'LOGS/history_with_magnitudes.csv'
    history.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")

    # Plot magnitudes against wavelengths
    plot_magnitudes(all_filter_results, output_file)
    
    
    
def synth_main(input_csv='synth_input.csv'):
    resolved_paths = load_and_resolve_paths('dir_inlist.txt')

    # Extract variables from resolved_paths
    base_dir = resolved_paths['base_dir']
    stellar_model = resolved_paths['stellar_model']
    instrument = resolved_paths['instrument']
    vega_sed_file = resolved_paths['vega_sed_file']
    output_file = os.path.join(base_dir, 'synth_output_with_magnitudes.csv')
    plot_synth(output_file)

    # Load Vega SED file
    vega_sed = pd.read_csv(vega_sed_file, sep=',', names=['wavelength', 'flux'], comment='#')
    print(f"Starting synthetic process...")

    # Load lookup table
    lookup_table_file = os.path.join(stellar_model, 'lookup_table.csv')
    lookup_table = pd.read_csv(lookup_table_file)
    print(f"Lookup table loaded: {lookup_table_file}")

    # Read synthetic input CSV
    synth_data = pd.read_csv(input_csv)
    print(f"Synthetic input data loaded: {input_csv}")

    bolometric_magnitudes = []
    all_filter_results = []

    print(f"Processing {len(synth_data)} models from synthetic input data...")

    for idx, row in synth_data.iterrows():
        print(f"Processing synthetic model {idx + 1}/{len(synth_data)}")

        # Extract necessary parameters
        teff = row.get('teff')
        log_g = row.get('logg')
        metallicity = row.get('meta', 0.0)  # Default metallicity if not provided
        wavelength = row.get('wavelength')
        flux = row.get('flux')

        if pd.isnull(teff) or pd.isnull(log_g) or pd.isnull(metallicity):
            print(f"Skipping model {idx + 1} due to missing parameters.")
            bolometric_magnitudes.append(None)
            all_filter_results.append(None)
            continue

        # Interpolate SED using teff, log_g, and lookup table
        model_wavelength, model_flux = interpolate_sed(teff, log_g, stellar_model, lookup_table)

        # Calculate bolometric magnitude
        bol_mag = calculate_bolometric_magnitude(model_wavelength, model_flux)
        bolometric_magnitudes.append(bol_mag)

        # Convolve model SED with filters
        filter_results = convolve_with_filter(model_wavelength, model_flux, instrument, vega_sed)
        all_filter_results.append(filter_results)

        # Print some info about the model
        print(f"Bolometric magnitude: {bol_mag}")
        print(f"Filter results (mags): {[mag for mag, _, _, _, _, _ in filter_results]}")

    # Add bolometric magnitudes to synthetic data
    synth_data['bolometric_magnitude'] = bolometric_magnitudes

    # Add filter results as separate columns in the DataFrame
    for idx, filter_results in enumerate(all_filter_results):
        if filter_results is None:
            continue
        for mag, flux_nu, flux_lambda, lambda_eff, peak_wavelength, filter_name in filter_results:
            # Column for magnitude
            mag_col_name = f"mag_{filter_name}"
            if mag_col_name not in synth_data:
                synth_data[mag_col_name] = np.nan
            synth_data.at[idx, mag_col_name] = mag

            # Column for flux_nu
            flux_nu_col_name = f"flux_nu_{filter_name}"
            if flux_nu_col_name not in synth_data:
                synth_data[flux_nu_col_name] = np.nan
            synth_data.at[idx, flux_nu_col_name] = flux_nu

            # Column for flux_lambda
            flux_lambda_col_name = f"flux_lambda_{filter_name}"
            if flux_lambda_col_name not in synth_data:
                synth_data[flux_lambda_col_name] = np.nan
            synth_data.at[idx, flux_lambda_col_name] = flux_lambda

            # Column for lambda_eff
            lambda_eff_col_name = f"lambda_eff_{filter_name}"
            if lambda_eff_col_name not in synth_data:
                synth_data[lambda_eff_col_name] = np.nan
            synth_data.at[idx, lambda_eff_col_name] = lambda_eff

    # Save results
    output_file = os.path.join(base_dir, 'synth_output_with_magnitudes.csv')
    synth_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")

    plot_synth(output_file)
    
    
def plot_synth(output_file):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Load the data
    data = pd.read_csv(output_file)

    # Calculate absolute and percentage differences
    data['flux_diff'] = data['flux'] - data['flux_lambda_g']
    data['flux_percent_diff'] = ((data['flux'] - data['flux_lambda_g']).abs() / data['flux_lambda_g']) * 100

    # Create pivot tables for the heatmaps
    pivot_table_abs = data.pivot_table(index='teff', columns='logg', values='flux_diff', aggfunc='mean')
    pivot_table_pct = data.pivot_table(index='teff', columns='logg', values='flux_percent_diff', aggfunc='mean')

    # Set up the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0})

    # Absolute difference heatmap
    im1 = axs[0].imshow(pivot_table_abs, origin='lower', aspect='auto', cmap='viridis', extent=[
        data['logg'].min(), data['logg'].max(),
        data['teff'].min(), data['teff'].max()
    ])
    
    axs[0].grid(color='gray', linestyle='--', linewidth=0.5)
    cbar1 = fig.colorbar(im1, ax=axs[0], location='left', pad=0.22)


    # Percentage difference heatmap
    im2 = axs[1].imshow(pivot_table_pct, origin='lower', aspect='auto', cmap='plasma', extent=[
        data['logg'].min(), data['logg'].max(),
        data['teff'].min(), data['teff'].max()
    ])

    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[1].grid(color='gray', linestyle='--', linewidth=0.5)
    cbar2 = fig.colorbar(im2, ax=axs[1], location='right', pad=0.01)


    cbar1.set_label(r'Absolute Flux Difference (SVO flux $-$ This flux)')
    cbar2.set_label(r'Percentage Difference (\%)')


    # Remove the Y-axis on the right-hand plot
    axs[1].yaxis.set_visible(False)

    # Adjust X-axis ticks and range for the left-hand plot
    axs[0].set_xticks([0, 1, 2, 3, 4])  # Custom ticks for the X-axis
    axs[0].set_xlim(data['logg'].min(), data['logg'].max())  # Ensure the range matches

    # Increase label sizes
    axs[0].set_ylabel(r'$\mathit{T}_{\mathrm{eff}}$ (K)', fontsize=14)
    axs[0].set_xlabel(r'                      $\log(g)$', fontsize=14)


    plt.show()

# Example usage with a placeholder file name
# Replace 'synth_output_with_magnitudes.csv' with the actual path to your file
# plot_synth('synth_output_with_magnitudes.csv')

    
if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process data for stellar magnitudes.")
    parser.add_argument(
        '-s', '--synth',
        action='store_true',
        help="Run the synthetic data processing instead of the main program."
    )
    args = parser.parse_args()

    # Decide which function to run
    if args.synth:
        synth_main(input_csv='hres_Palomar.ZTF.g_photometry.csv')
        #synth_main(input_csv='Kurucz2003all_JWSTNIRCam.F480M_photometry.csv')        
        #synth_main(input_csv='Kurucz2003all_Liverpool.IOO.SDSS-r_photometry.csv')
    else:
        main()
