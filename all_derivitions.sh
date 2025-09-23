#!/bin/bash

# --- Configuration ---
# The name of your Python script
PYTHON_SCRIPT="1word1mask.py"
# The directory where your input files are stored
INPUT_DIR="deasy"
# The directory where you want to save the results
OUTPUT_DIR="results"
# Optional arguments for the python script (e.g., "-s" for scores).
# The number of candidates (-n) is now set automatically based on the input file's line count.
PYTHON_ARGS=""

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
        # Count the number of lines in the input file to use as the candidate count
        num_lines=$(wc -l < "$input_file")
        
        echo "Running: ${template} with ${input_file} (${num_lines} words)..."
        # Run the python script with -n set to the number of lines
        python "$PYTHON_SCRIPT" "$input_file" "$template" -n "$num_lines" $PYTHON_ARGS > "$output_file"
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
        # Count the number of lines in the input file to use as the candidate count
        num_lines=$(wc -l < "$input_file")
        
        echo "Running: ${template} with ${input_file} (${num_lines} words)..."
        # Run the python script with -n set to the number of lines
        python "$PYTHON_SCRIPT" "$input_file" "$template" -n "$num_lines" $PYTHON_ARGS > "$output_file"
    else
        echo "Warning: Input file not found at ${input_file}, skipping ${template}."
    fi
done

echo "All templates processed successfully. ✅"