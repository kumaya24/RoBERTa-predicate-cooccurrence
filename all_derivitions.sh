#!/bin/bash

# --- Configuration ---
# The name of your Python script
PYTHON_SCRIPT="1word1mask.py"
# The directory where your input files are stored
INPUT_DIR="deasy"
# The directory where you want to save the results
OUTPUT_DIR="results"
# Optional arguments for the python script (e.g., "-n 10" or "-n 5 -s")
PYTHON_ARGS="-n 2000"

# --- Template Categories ---
# List of templates that use the intransitive verb list
vintran_templates=(
    "nom_vintran" "agent_vintran" "evt_vintran" "participleAdj_vintran"
    "causative_vintran" "res_vintran" "item_vintran" "state_vintran" "inst_vintran"
)

# List of templates that use the transitive verb list
vtran_templates=(
    "nom_vtran" "agent_vtran" "evt_vtran" "participleAdj_vtran"
    "causative_vtran" "res_vtran" "item_vtran" "state_vtran" "inst_vtran"
)

# --- Main Script Logic ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Processing intransitive templates..."
# Loop through each intransitive template
for template in "${vintran_templates[@]}"; do
    input_file="${INPUT_DIR}/vintrans.txt"
    output_file="${OUTPUT_DIR}/${template}.txt"

    # Check if the input file exists before running
    if [ -f "$input_file" ]; then
        echo "Running: ${template} with ${input_file}"
        python "$PYTHON_SCRIPT" "$input_file" "$template" $PYTHON_ARGS > "$output_file"
    else
        echo "Warning: Input file not found at ${input_file}, skipping ${template}."
    fi
done

echo "Processing transitive templates..."
# Loop through each transitive template
for template in "${vtran_templates[@]}"; do
    input_file="${INPUT_DIR}/vtrans.txt"
    output_file="${OUTPUT_DIR}/${template}.txt"

    # Check if the input file exists before running
    if [ -f "$input_file" ]; then
        echo "Running: ${template} with ${input_file}"
        python "$PYTHON_SCRIPT" "$input_file" "$template" $PYTHON_ARGS > "$output_file"
    else
        echo "Warning: Input file not found at ${input_file}, skipping ${template}."
    fi
done

echo "All templates processed successfully."
