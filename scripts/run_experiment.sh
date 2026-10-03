#!/bin/bash

set -e
trap 'echo "Error: the script failed on line $LINENO."' ERR

# Arguments
mode="${1,,}"       # train|eval       
task="${2,,}"       # ps|pbc
experiment="${3,,}" # ove|com|rec
run="$4"            # 1|2|3

# Validate arguments
if [[ "$mode" != "train" && "$mode" != "eval" ]] ||
   [[ "$task" != "pbc" && "$task" != "ps" ]] ||
   [[ "$experiment" != "ove" && "$experiment" != "com" && "$experiment" != "rec" ]] ||
   [[ "$run" != "1" && "$run" != "2" && "$run" != "3" ]]; then
    echo "Error: invalid arguments."
    echo "Usage: bash scripts/run_experiment.sh train|eval pbc|ps ove|com|rec 1|2|3"
    exit 1
fi

# Run experiment
python src/main.py --mode "$mode" --task "$task" --experiment "$experiment" --run "$run"