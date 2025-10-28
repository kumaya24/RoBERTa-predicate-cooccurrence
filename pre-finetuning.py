import torch
from transformers import RobertaTokenizer
from typing import List, Dict, Tuple
import random
import os
import csv
import argparse
from nom_prompts import TEMPLATE_OPTIONS # Assuming you are using the separate prompt file

# --- Configuration ---
TSV_FILE_PATH = "nominalization_pairs.tsv"
MAX_SEQ_LENGTH = 128 

# --- Argument Parsing ---
# NOTE: The argument parser is minimal here since its primary purpose is
# to get the template_option needed for the dataset file name.

parser = argparse.ArgumentParser(description="Generate MLM fine-tuning dataset with single-token nominalizations.")

parser.add_argument("template_option", choices=list(TEMPLATE_OPTIONS.keys()),
                       help="Choose a template option (e.g. nom_vintran)")

args = parser.parse_args()

TEMPLATE_KEY_TO_USE = args.template_option 

# --- Model and Tokenizer Setup ---
# Initialize tokenizer once for use in filtering and dataset creation
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# --- Data Loading Utility ---

def load_and_reorder_pairs(input_filename: str) -> Dict[str, str]:
    """Loads pairs from TSV (Nominalization, Verb) and returns {Verb: Nominalization}."""
    reordered_pairs = {}
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None) # Skip header
            for row in reader:
                if len(row) == 2:
                    nominalization = row[0].strip()
                    verb = row[1].strip()
                    reordered_pairs[verb] = nominalization
        print(f"INFO: Successfully loaded {len(reordered_pairs)} pairs from {input_filename}.")
        return reordered_pairs
    except FileNotFoundError:
        print(f"WARNING: The file '{input_filename}' was not found. Using empty data.")
        return {}
    except Exception as e:
        print(f"WARNING: Error loading TSV: {e}. Using empty data.")
        return {}

# Load the nominalization pairs
NOMINALIZATION_DEMOS = load_and_reorder_pairs(TSV_FILE_PATH)


# --- Core Dataset Creation Logic (WITH THE SINGLE-TOKEN FILTER) ---

def create_mlm_dataset(
    template_key: str, 
    template_options: Dict[str, List[str]], 
    nominalization_pairs: Dict[str, str],
    tokenizer: RobertaTokenizer
) -> torch.utils.data.Dataset:
    """
    Transforms Verb-Noun pairs into tokenized MLM inputs and labels, 
    filtering for single-token nominalizations only.
    """
    
    templates = template_options.get(template_key)
    if not templates:
        print(f"ERROR: Template key '{template_key}' not found.")
        return None

    selected_template = random.choice(templates)
    print(f"INFO: Using template: '{selected_template}' for dataset generation.")
    
    mlm_examples = []
    skipped_count = 0 
    
    for verb, target_noun in nominalization_pairs.items():
        # 1. Check Token Count (THE FIX: MUST BE ONE TOKEN)
        # add_special_tokens=False is critical here
        target_token_ids = tokenizer.encode(target_noun, add_special_tokens=False)
        
        # If the nominalization is NOT exactly one token, skip the pair.
        if len(target_token_ids) != 1:
            skipped_count += 1
            continue 

        # 2. Create the prompt with the mask and verb
        prompt = selected_template.replace("{w}", verb).replace("<mask>", tokenizer.mask_token)
        
        # 3. Tokenize the input prompt
        tokenized_input = tokenizer(
            prompt, 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_SEQ_LENGTH,
            return_tensors='pt'
        )
        
        input_ids = tokenized_input['input_ids'][0]
        
        # 4. Create the target labels (MLM Loss Labels)
        labels = torch.full(input_ids.shape, -100, dtype=torch.long)
        
        # a. Find the index of the <mask> token
        mask_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        
        if mask_indices.numel() == 0:
            continue
            
        mask_idx = mask_indices[0].item()
        
        # b. Set the label (We KNOW len(target_token_ids) is 1)
        labels[mask_idx] = target_token_ids[0]
        
        mlm_examples.append({
            'input_ids': tokenized_input['input_ids'].squeeze(),
            'attention_mask': tokenized_input['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
        })

    print(f"INFO: Skipped {skipped_count} pairs because the nominalization was multi-token.")
    print(f"INFO: Using {len(mlm_examples)} pairs for fine-tuning dataset.")

    # Convert the list of dicts to PyTorch TensorDataset
    if not mlm_examples:
        return None
        
    all_input_ids = torch.stack([ex['input_ids'] for ex in mlm_examples])
    all_attention_masks = torch.stack([ex['attention_mask'] for ex in mlm_examples])
    all_labels = torch.stack([ex['labels'] for ex in mlm_examples])

    final_dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_masks, all_labels)
    return final_dataset

# --- Execute Dataset Creation and Save ---

mlm_dataset = create_mlm_dataset(
    TEMPLATE_KEY_TO_USE, 
    TEMPLATE_OPTIONS, 
    NOMINALIZATION_DEMOS,
    tokenizer # Pass the global tokenizer
)

if mlm_dataset:
    save_filename = f"{TEMPLATE_KEY_TO_USE}_mlm_finetune_dataset.pt"
    # Ensure this part of the script is run on the CPU if dealing with large data/low VRAM
    # but for saving, device doesn't matter much.
    torch.save(mlm_dataset, save_filename)
    print(f"\n✅ Dataset successfully created and saved to '{save_filename}'.")
    print(f"Ready for fine-tuning {len(mlm_dataset)} examples with RobertaForMaskedLM.")