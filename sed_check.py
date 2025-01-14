import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load the CSV file into a DataFrame
csv_file = "full_arrays.csv"  # Replace with the actual path if needed
data = pd.read_csv(csv_file)

# Step 1: Check data types
print("Data types before cleaning:")
print(data.dtypes)

# Step 2: Convert 'Wavelengths' to numeric (if needed)
data['Wavelengths'] = pd.to_numeric(data['Wavelengths'], errors='coerce')

# Step 3: Drop rows with NaN in 'Wavelengths' or 'Convolved Fluxes'
data = data.dropna(subset=['Wavelengths', 'Convolved Fluxes'])

# Step 4: Check data after cleaning
print("\nData sample after cleaning:")
print(data.head())

print("\nData types after cleaning:")
print(data.dtypes)

# Step 5: Plot the data
plt.figure(figsize=(10, 6))

# Plot Convolved Fluxes as points without connecting lines
#plt.scatter(data['Wavelengths'], data['Convolved Fluxes'], label="Convolved Fluxes")
plt.scatter(data['Convolved Fluxes'], data['Wavelengths'], label="Convolved Fluxes")
#plt.scatter(data['Wavelengths'], data['Fluxes'], label="Convolved Fluxes")
plt.scatter(data['Convolved Fluxes'], data['Fluxes'], label="Convolved Fluxes")
#plt.scatter(data['Wavelengths'], data['Interpolated Filter'], label="Convolved Fluxes")

# Add labels, title, and grid
plt.xlabel('Wavelengths')
plt.ylabel('Convolved Fluxes')
plt.title('Array Data Visualization')

# Limit number of ticks on the x-axis for clarity
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

# Add legend and adjust layout
plt.legend()
plt.tight_layout()
plt.show()

