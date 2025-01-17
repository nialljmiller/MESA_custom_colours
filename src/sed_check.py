import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file = "full_arrays.csv"  # Replace with the actual path if needed
data = pd.read_csv(csv_file)
print(data)
# Plot the data
plt.figure(figsize=(10, 6))

# Plot each column (other than 'Index') on the same plot
plt.plot(data['Index'], data['Wavelengths'], label='Wavelengths', linestyle='-', marker='o')
plt.plot(data['Index'], data['Fluxes'], label='Fluxes', linestyle='--', marker='x')
plt.plot(data['Index'], data['Convolved Fluxes'], label='Convolved Fluxes', linestyle='-.', marker='^')
plt.plot(data['Index'], data['Interpolated Filter'], label='Interpolated Filter', linestyle=':', marker='s')

# Add labels, legend, and grid
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Array Data Visualization')
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

