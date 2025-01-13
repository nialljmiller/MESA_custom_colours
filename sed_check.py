import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file = "full_arrays.csv"  # Replace with the actual path if needed
data = pd.read_csv(csv_file)
print(data)

# Plot the data
plt.figure(figsize=(10, 6))

# Plot Convolved Fluxes as points without connecting lines
plt.plot(data['Wavelengths'], data['Convolved Fluxes'], label='Convolved Fluxes', linestyle='', marker='o')
plt.plot(data['Wavelengths'], data['Fluxes'], label='Fluxes', linestyle='', marker='x')

# Add labels, legend, and grid
plt.xlabel('Wavelengths')
plt.ylabel('Convolved Fluxes')
plt.title('Array Data Visualization')
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

