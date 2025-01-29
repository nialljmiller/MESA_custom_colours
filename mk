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

# Proceed to make
cd make
make
check_okay
