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
REMOTE_URL="https://nialljmiller.com/media/bigdata/MESA/testdata.txz"
LOCAL_FILE="testdata.txz"

# Download the file if it doesn't exist locally
if [ ! -f "$LOCAL_FILE" ]; then
	echo "Downloading test data from $REMOTE_URL..."
	curl -L -o "$LOCAL_FILE" "$REMOTE_URL" || { echo "Download failed"; exit 1; }
else
	echo "Test data file already exists locally. Skipping download."
fi

# Extract and merge the contents into data/
if [ -f "$LOCAL_FILE" ]; then
	echo "Extracting test data..."
	tar -xJf "$LOCAL_FILE" --directory=data --strip-components=1 || { echo "Extraction failed"; exit 1; }
else
	echo "Missing test data file. Extraction skipped."
fi

# Proceed to make
cd make
make
check_okay
