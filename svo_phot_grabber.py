import requests
import os
import csv
import re

# Base URL for photometry data
phot_base_url = "http://svo2.cab.inta-csic.es/theory/newov2/phot.php"

# Output directory
output_dir = "data/synthetic_photometry/"
os.makedirs(output_dir, exist_ok=True)

def parse_photometry(file_content):
    """
    Parse photometry data and extract metadata along with filter, wavelength, and flux.
    
    Args:
        file_content (str): The content of the photometry file.

    Returns:
        dict: A dictionary with parsed metadata and photometry data.
    """
    metadata = {}
    photometry_data = {}
    
    # Split content into lines
    lines = file_content.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#') and '=' in line:
            # Extract key-value pairs from metadata
            key, value = line.split('=', 1)
            key = key.strip('#').strip()
            value = value.split('(')[0].strip()  # Remove parentheses and text inside
            metadata[key] = value
        elif not line.startswith('#') and line:  # Non-comment line with data
            # Extract photometry data
            parts = re.split(r'\s+', line)
            if len(parts) == 3:
                photometry_data = {
                    "filter": parts[0],
                    "wavelength": parts[1],
                    "flux": parts[2]
                }

    # Combine metadata and photometry data
    return {**metadata, **photometry_data}

def download_photometry_data(model, filter_name, start_id, output_csv):
    """
    Download all photometry data for a given model and filter, and save it to a CSV.

    Args:
        model (str): The stellar model (e.g., 'Kurucz2003all').
        filter_name (str): The filter name (e.g., 'JWST/NIRCam.F480M').
        start_id (int): The starting ID for the photometry data.
        output_csv (str): Path to the output CSV file.
    """
    data_rows = []
    current_id = start_id

    while True:
        # Construct request URL
        params = {
            "model": model,
            "fid": current_id,
            "filter": filter_name,
            "format": "ascii"
        }
        response = requests.get(phot_base_url, params=params)

        # Check if the response is valid
        if response.status_code == 200 and len(response.content) > 1024:
            # Parse the response content
            content = response.content.decode("utf-8")
            parsed_data = parse_photometry(content)
            data_rows.append(parsed_data)
            print(f"Downloaded: model={model}, fid={current_id}, filter={filter_name}")
            current_id += 1
        else:
            print(f"No more data available for model={model}, filter={filter_name}. Last ID: {current_id - 1}")
            break

    # Write all data to CSV
    if data_rows:
        with open(output_csv, mode='w', newline='') as csv_file:
            fieldnames = sorted(data_rows[0].keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_rows)
        print(f"Data saved to {output_csv}")
    else:
        print("No data to write.")

# Example usage
model = "Kurucz2003all"
filter_name = "JWST/NIRCam.F480M"
start_id = 16375
output_csv = os.path.join(output_dir, f"{model}_{filter_name}_photometry.csv")

download_photometry_data(model, filter_name, start_id, output_csv)

