import re
import sys
import os

def extract_nominalization_pairs(file_path: str):
    pattern = re.compile(r':ORTH\s*"([^"]+?)".*?:VERB\s*"([^"]+?)"', re.DOTALL)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The input file '{file_path}' was not found. Please ensure the path is correct.")
        return
    except Exception as e:
        print(f"An error occurred while reading the input file: {e}")
        return

    # Find all matches
    matches = pattern.findall(content)
    
    if not matches:
        print("No nominalization pairs found using the expected format.")
        return

    output_file_name = "nominalization_pairs.tsv"
    
    output_path = os.path.join(os.path.dirname(file_path), output_file_name) if os.path.dirname(file_path) else output_file_name

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # Write TSV header
            outfile.write("Nominalization\tVerb\n")
            
            # Write each matched pair
            for noun, verb in matches:
                # Use tab (\t) delimiter and newline (\n) for TSV format
                outfile.write(f"{noun.strip()}\t{verb.strip()}\n")
        
        print(f"--- Extraction complete ---")
        print(f"Successfully extracted {len(matches)} pairs.")
        print(f"Results saved to '{output_path}'.")

    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract.py <path_to_NOMLEX-2001.exp_file>")
        print("\nNote: Make sure 'extract.py' itself is either in your current directory or referenced using its full path.")
    else:
        # The file path is the first argument after the script name
        file_path = sys.argv[1]
        extract_nominalization_pairs(file_path)
