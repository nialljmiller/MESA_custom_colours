import os
from astroquery.svo_fps import SvoFps
from astropy import units as u
from astropy.table import vstack, unique
import pandas as pd

# Set a longer timeout
SvoFps.TIMEOUT = 300  # Increase timeout to 5 minutes

def fetch_facility_links(filters_df):
    """
    Allow the user to select specific facilities from the list.
    """
    # Get unique facilities
    facilities = filters_df['Facility'].unique()

    # Display facilities for user selection
    print("\nAvailable facilities:")
    for idx, facility in enumerate(facilities, start=1):
        print(f"{idx}. {facility}")

    # Ask the user to select facilities
    user_input = input("Enter the numbers of the facilities you want to select, separated by commas: ")

    try:
        selected_indices = [int(num.strip()) - 1 for num in user_input.split(",")]
        selected_facilities = [facilities[i] for i in selected_indices if 0 <= i < len(facilities)]
        return filters_df[filters_df['Facility'].isin(selected_facilities)]
    except (ValueError, IndexError):
        print("Invalid input. Please enter valid numbers separated by commas.")
        return pd.DataFrame()

def main():
    # Define wavelength ranges for different spectral regions (in Angstroms)
    wavelength_ranges = [
        ('X-ray', 0.1 * u.AA, 100 * u.AA),
        ('UV', 100 * u.AA, 4000 * u.AA),
        ('Optical', 4000 * u.AA, 7000 * u.AA),
        ('NIR', 7000 * u.AA, 25000 * u.AA),
        ('MIR', 25000 * u.AA, 250000 * u.AA),
        ('FIR', 250000 * u.AA, 1e7 * u.AA),
        ('Radio', 1e7 * u.AA, 1e8 * u.AA),
    ]

    # List to hold all filters
    all_filters = []

    # Loop over each wavelength range
    for region_name, wavelength_min, wavelength_max in wavelength_ranges:
        print(f"\nRetrieving filters for {region_name} range ({wavelength_min} - {wavelength_max})")
        try:
            filters_table = SvoFps.get_filter_index(
                wavelength_eff_min=wavelength_min, wavelength_eff_max=wavelength_max
            )
            print(f"Retrieved {len(filters_table)} filters for {region_name}")
            all_filters.append(filters_table)
        except Exception as e:
            print(f"Error retrieving filters for {region_name}: {e}")

    # Combine all filters into a single table
    if all_filters:
        combined_filters_table = vstack(all_filters)
        # Remove duplicate filters based on 'filterID'
        combined_filters_table = unique(combined_filters_table, keys='filterID')
    else:
        print("No filters retrieved. Exiting.")
        return

    # Convert to Pandas DataFrame
    filters_df = combined_filters_table.to_pandas()

    # Print total number of filters
    print(f"\nTotal number of filters to process: {len(filters_df)}")

    # Fetch user-selected filters by facility
    selected_filters_df = fetch_facility_links(filters_df)
    if selected_filters_df.empty:
        print("No filters selected. Exiting.")
        return

    # Base directory to save filters
    base_dir = '~/mesa/star/test_suite/custom_colors/data/filters'
    base_dir = os.path.expanduser(base_dir)

    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Loop over each selected filter
    for idx, row in selected_filters_df.iterrows():
        filter_id = row['filterID']
        facility = row.get('Facility', 'UnknownFacility')
        instrument = row.get('Instrument', 'UnknownInstrument')
        filter_name = row.get('Band', filter_id)

        if filter_name == '':
            filter_name = filter_id.split('.')[-1]

        # Print progress
        print(f"\nProcessing filter {idx + 1}/{len(selected_filters_df)}: {filter_id}")

        # Clean up facility and filter names to make valid directory/file names
        facility_dir = ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in facility).strip()
        instrument_dir = ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in instrument).strip()
        filter_filename = ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filter_name).strip() + '.dat'

        # Full path for the facility and instrument directory
        directory_path = os.path.join(base_dir, facility_dir, instrument_dir)

        # Create directories if they don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Download transmission data
        try:
            transmission_data = SvoFps.get_transmission_data(filter_id)
            if transmission_data is not None and len(transmission_data) > 0:
                # Save the data to a file
                file_path = os.path.join(directory_path, filter_filename)
                transmission_data.write(file_path, format='ascii.csv', overwrite=True)
                print(f"Saved filter '{filter_name}' to {file_path}")
            else:
                print(f"No transmission data for filter '{filter_id}'")
        except Exception as e:
            print(f"Error downloading data for filter '{filter_id}': {e}")

if __name__ == "__main__":
    main()

