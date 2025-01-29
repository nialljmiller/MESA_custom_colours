import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_output_files(directory):
    """Finds all files in the directory that contain 'SED.csv' in their name."""
    return [f for f in os.listdir(directory) if 'SED.csv' in f and os.path.isfile(os.path.join(directory, f))]

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def process_all_files(directory, xlim=None):
    output_files = find_output_files(directory)
    if not output_files:
        print("No output files found.")
        return

    plt.figure(figsize=(12, 6))

    full_sed_plotted = False

    for file_path in output_files:
        df = pd.read_csv(os.path.join(directory, file_path), delimiter=',', header=0).rename(columns=str.strip).dropna()

        # Assign columns based on their names
        wavelengths = df["wavelengths"].to_numpy()
        flux = df["fluxes"].to_numpy()
        convolved_flux = df["convolved_flux"].to_numpy()
        filter_wavelengths = df["filter_wavelengths"].to_numpy()
        filter_trans = df["filter_trans"].to_numpy()

        print(f"Processing {file_path}")
        print("Wavelengths shape:", wavelengths.shape)
        print("Flux shape:", flux.shape)
        print("Convolved Flux shape:", convolved_flux.shape)

        # Normalize data
        nflux = normalize(flux)
        nconvolved_flux = normalize(convolved_flux)
        nfilter_trans = normalize(filter_trans)

        # Plot full SED only once (assume it is the same across files)
        if not full_sed_plotted:
            plt.plot(wavelengths, flux, label="Full SED", linewidth=1, color="black")
            full_sed_plotted = True

        # Plot convolved SED
        plt.plot(wavelengths, convolved_flux, label=f"Convolved SED ({file_path})", linewidth=1)

        # Plot normalized filters
        plt.plot(filter_wavelengths, nfilter_trans, label=f"Filter ({file_path})", linewidth=1, linestyle="--")

    # Formatting
    plt.xlabel("Wavelengths")
    plt.ylabel("Flux")
    #plt.legend()
    plt.title("Combined SEDs and Filters")
    plt.ticklabel_format(style='plain', useOffset=False)

    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    plt.tight_layout()
    plt.show()


def main():
    directory = '../LOGS/SED/'  # Change if needed
    process_all_files(directory, xlim=[0,0.0005])

if __name__ == "__main__":
    main()
