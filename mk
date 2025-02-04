#!/bin/bash

function check_okay {
	if [ $? -ne 0 ]
	then
		echo
		echo "FAILED"
		echo
		exit 1
	fi
}

# Define directories to be created
DIRS=(
	"LOGS"
	"LOGS/SED"
	"data/filters"
	"data/MIST"
	"data/stellar_models"
)

# Create directories if they don't exist
for dir in "${DIRS[@]}"; do
	if [ ! -d "$dir" ]; then
		mkdir -p "$dir"
	fi
done

# Define remote data file location
REMOTE_FILE="nillmill.ddns.net:/media/bigdata/MESA/testdata.txz"
LOCAL_FILE="testdata.txz"
EXTRACTED_FLAG="data/extracted_marker"  # Marker file to track extraction

# Download the file if it doesn't exist locally
if [ ! -f "$LOCAL_FILE" ]; then
	echo "Downloading test data from $REMOTE_FILE..."
	scp "$REMOTE_FILE" "$LOCAL_FILE" || { echo "Download failed"; exit 1; }
else
	echo "Test data file already exists locally. Skipping download."
fi

# Extract and merge the contents into data/ only if it hasn't been extracted
if [ ! -f "$EXTRACTED_FLAG" ]; then
	if [ -f "$LOCAL_FILE" ]; then
		echo "Extracting test data..."
		tar -xJf "$LOCAL_FILE" --directory=data --strip-components=1 || { echo "Extraction failed"; exit 1; }
		touch "$EXTRACTED_FLAG"  # Create marker file
	else
		echo "Missing test data file. Extraction skipped."
	fi
else
	echo "Test data already extracted. Skipping extraction."
fi

# Proceed to make
cd make
make
check_okay
