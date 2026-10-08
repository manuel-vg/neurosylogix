#!/bin/bash

set -e
trap 'echo "Error: the script failed on line $LINENO."' ERR

# Construct proofs using symbolic and hybrid models
python src/construct_proofs.py

# Plot results
python src/plot_results.py