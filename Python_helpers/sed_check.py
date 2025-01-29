import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
# Load the CSV file into a DataFrame

def clip_data(wavelengths, flux, convolved_flux, lower_percentile=5, upper_percentile=95):
    valid_flux_indices = np.where(flux > 0)[0]  # Ignore zero flux
    valid_flux = flux[valid_flux_indices]  # Only use non-zero flux for stats
    lower_threshold = np.percentile(valid_flux, lower_percentile)
    upper_threshold = np.percentile(valid_flux, upper_percentile)

    mask = (flux >= lower_threshold) & (flux <= upper_threshold)
    
    first_valid = np.argmax(mask)
    last_valid = len(mask) - np.argmax(mask[::-1]) - 1

    contiguous_mask = np.arange(first_valid, last_valid + 1)

    return (wavelengths[contiguous_mask],flux[contiguous_mask],convolved_flux[contiguous_mask])

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

import pandas as pd
import numpy as np

# Read CSV while ignoring rows with missing values
df = pd.read_csv('output.csv', delimiter=',', header=0)

# Drop rows with missing values to avoid format errors
df = df.dropna()

# Convert to numpy arrays
wavelengths = df.iloc[:, 0].to_numpy()
flux = df.iloc[:, 1].to_numpy()
#filter_wavelengths = df.iloc[:, 2].to_numpy()
#filter_trans = df.iloc[:, 3].to_numpy()
convolved_flux = df.iloc[:, 2].to_numpy()
#interpolated_filter = df.iloc[:, 5].to_numpy()

# Print shapes to verify
print("Wavelengths shape:", wavelengths.shape)
print("Fluxes shape:", flux.shape)
#print("Filter Wavelengths shape:", filter_wavelengths.shape)
#print("Filter Transmission shape:", filter_trans.shape)
print("Convolved Flux shape:", convolved_flux.shape)
#print("Interpolated Filter shape:", interpolated_filter.shape)


# Plot the normalized data
plt.scatter(wavelengths, convolved_flux, label="Convolved Fluxes")
plt.scatter(wavelengths, flux, label="RAW Fluxes (Non-Zero)")
#plt.scatter(filter_wavelengths, filter_trans, label="Filter")
plt.xlabel("Wavelengths")
plt.ylabel("Flux")
plt.legend()
plt.show()


nflux = normalize(flux)
nconvolved_flux = normalize(convolved_flux)
#nfilter_trans = normalize(filter_trans)


# Plot the normalized data
plt.scatter(wavelengths, nconvolved_flux, label="Convolved Fluxes")
plt.scatter(wavelengths, nflux, label="RAW Fluxes (Non-Zero)")
#plt.scatter(filter_wavelengths, nfilter_trans, label="Filter")
plt.xlabel("Wavelengths")
plt.ylabel("Flux")
plt.legend()
plt.show()







clipped_wavelengths, clipped_flux, clipped_convolved_flux = clip_data(wavelengths, flux, convolved_flux, lower_percentile=10, upper_percentile=100)

plt.scatter(clipped_wavelengths, clipped_flux, label="Clipped Flux")
plt.scatter(clipped_wavelengths, clipped_convolved_flux, label="Clipped Convolved Flux")
plt.xlabel("Wavelengths")
plt.ylabel("Flux")
plt.legend()
plt.show()




def binned_trapz_integral(wavelengths, flux, num_bins):
    """
    Bin the data into `num_bins` and compute the trapezoidal integral of the binned data.

    Parameters:
        wavelengths (array-like): Wavelength values.
        flux (array-like): Flux values corresponding to wavelengths.
        num_bins (int): Number of bins to group the data into.

    Returns:
        float: Integrated flux of the binned data.
        tuple: Binned wavelengths and flux arrays.
    """
    # Create bins and find the bin centers
    bins = np.linspace(wavelengths.min(), wavelengths.max(), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Bin the data
    binned_flux = np.zeros(num_bins)
    for i in range(num_bins):
        # Find indices of points within the current bin
        bin_mask = (wavelengths >= bins[i]) & (wavelengths < bins[i + 1])
        if bin_mask.sum() > 0:
            # Average the flux in the bin
            binned_flux[i] = flux[bin_mask].mean()
        else:
            # Handle empty bins
            binned_flux[i] = 0

    # Compute the trapezoidal integral
    integrated_flux = np.trapz(binned_flux, bin_centers)
    return integrated_flux, bin_centers, binned_flux

nbins_list = []
nflux_list = []
# Test the function with different bin counts
num_bins_list = np.linspace(1, 5, 500)
for num_bins in num_bins_list:
	num_bins = int(len(clipped_convolved_flux)/num_bins)
	integrated_flux, binned_wavelengths, binned_flux = binned_trapz_integral(clipped_wavelengths, clipped_convolved_flux, num_bins)
	#print(f"Num bins: {num_bins}, Integrated Flux: {integrated_flux:.4f}")
	nbins_list.append(num_bins)
	nflux_list.append(integrated_flux)
	# Plot the binned data

plt.clf()
plt.scatter(nbins_list, nflux_list)
plt.xlabel("num bins")
plt.ylabel("integrated Flux")
plt.legend()
plt.show()