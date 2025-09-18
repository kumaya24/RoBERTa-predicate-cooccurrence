#!/bin/bash

# --- Configuration ---
# The name of your python script
PYTHON_SCRIPT="predicates_from_roberta.py"
# The directory to save output files in
OUTPUT_DIR="deasy"
# Optional arguments for the python script (e.g., "-n 2000" or "-n 50 -s")
PYTHON_ARGS="-n 2000"

# --- List of all templates from your script ---
templates=(
    "adj" "noun" "vintrans" "vtrans" "particleVerbUp" "particleVerbDown"
    "particleVerbIn" "particleVerbOut" "particleVerbOver" "particleVerbAlong"
    "particleVerbAway" "particleVerbTo" "particleVerbInto" "particleVerbOff"
    "particleVerbOn" "particleVerbAround" "particleVerbUnder" "prep" "compadj"
    "adv" "aux" "determiner" "pronoun" "conj" "interjection" "negation"
    "superlative" "complementClauseVerb" "complementClauseNoun"
    "nullComplementClauseVerb" "nullComplementClauseNoun" "baseComplementClauseVerb"
    "baseComplementClauseNoun" "infinitivalComplementClauseVerb"
    "infinitivalComplementClauseNoun" "raisingVerb" "gerundVerb"
)

# --- Main Script Logic ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting to process templates..."

# Loop through each template in the list
for template in "${templates[@]}"; do
    output_file="${OUTPUT_DIR}/${template}.txt"
    echo "Processing template: '${template}' -> saving to ${output_file}"
    
    # Run the python script and redirect its output to the file
    python "$PYTHON_SCRIPT" "$template" $PYTHON_ARGS > "$output_file"
done

echo "All templates processed successfully."
