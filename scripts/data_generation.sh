#!/bin/bash

set -e
trap 'echo "Error: the script failed on line $LINENO."' ERR

# Arguments
model="${1^^}"        # T5|GPT
task="${2,,}"         # ps|pbc
experiment="${3,,}"   # ove|com|rec

# Validate arguments
if [[ "$model" != "T5" && "$model" != "GPT" ]] ||
   [[ "$task" != "pbc" && "$task" != "ps" ]] ||
   [[ "$experiment" != "ove" && "$experiment" != "com" && "$experiment" != "rec" ]]; then
    echo "Error: invalid arguments."
    echo "Usage: bash scripts/data_generation.sh T5|GPT pbc|ps ove|com|rec"
    exit 1
fi

output="data/${model}_${task}_${experiment}"

# Generate datasets
python src/dataset_generation.py "$output" "$model" "$task" "$experiment"

echo "Dataset successfully created in: $output"