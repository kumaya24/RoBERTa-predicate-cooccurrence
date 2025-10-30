import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from typing import List, Dict, Tuple
import random
import os
import csv
import argparse
from nom_prompts import TEMPLATE_OPTIONS

TSV_FILE_PATH = "nominalization_pairs.tsv"
MAX_SEQ_LENGTH = 32 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Generate soft-label dataset for multi-token words.")
parser.add_argument("template_option", choices=list(TEMPLATE_OPTIONS.keys()),
                       help="Choose a template option (e.g. nom_vintran)")
args = parser.parse_args()
TEMPLATE_KEY_TO_USE = args.template_option 

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
baseline_model = RobertaForMaskedLM.from_pretrained("roberta-base").to(DEVICE)
baseline_model.eval() 
def load_and_reorder_pairs(input_filename: str) -> Dict[str, str]:
    reordered_pairs = {}
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) == 2:
                    nominalization = row[0].strip()
                    verb = row[1].strip()
                    reordered_pairs[verb] = nominalization
        print(f"Successfully loaded.")
        return reordered_pairs
    except FileNotFoundError:
        print(f"WARNING: The file '{input_filename}' was not found. Using empty data.")
        return {}
    except Exception as e:
        print(f"WARNING: Error loading TSV: {e}. Using empty data.")
        return {}

NOMINALIZATION_DEMOS = load_and_reorder_pairs(TSV_FILE_PATH)

def create_soft_label_dataset(
    template_key: str, 
    template_options: Dict[str, List[str]], 
    nominalization_pairs: Dict[str, str]
) -> torch.utils.data.Dataset:
    
    templates = template_options.get(template_key)
    if not templates:
        print(f"ERROR: Template key '{template_key}' not found.")
        return None

    mlm_examples = []
    
    for verb, target_noun in nominalization_pairs.items():
        for selected_template in templates:
        
            prompt = selected_template.replace("{w}", verb).replace("<mask>", tokenizer.mask_token)
            
            tokenized_input = tokenizer(
                prompt, 
                truncation=True, 
                padding="max_length", 
                max_length=MAX_SEQ_LENGTH,
                return_tensors='pt'
            ).to(DEVICE)
            
            mask_positions = (tokenized_input.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if mask_positions.numel() == 0:
                continue
            mask_idx = mask_positions[0].item()
            
            with torch.no_grad():
                target_logits = baseline_model(**tokenized_input).logits[0, mask_idx]
                
            target_token_id = tokenizer.encode(target_noun, add_special_tokens=False)[0] if tokenizer.encode(target_noun, add_special_tokens=False) else tokenizer.unk_token_id
            hard_labels = torch.full(tokenized_input.input_ids[0].shape, -100, dtype=torch.long)
            hard_labels[mask_idx] = target_token_id 

            mlm_examples.append({
                'input_ids': tokenized_input['input_ids'].squeeze().cpu(),
                'attention_mask': tokenized_input['attention_mask'].squeeze().cpu(),
                'soft_labels': target_logits.cpu(),  
                'hard_labels': hard_labels.squeeze().cpu()
            })
    
    if not mlm_examples:
        return None
        
    all_input_ids = torch.stack([ex['input_ids'] for ex in mlm_examples])
    all_attention_masks = torch.stack([ex['attention_mask'] for ex in mlm_examples])
    all_soft_labels = torch.stack([ex['soft_labels'] for ex in mlm_examples])
    all_hard_labels = torch.stack([ex['hard_labels'] for ex in mlm_examples])

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'soft_labels': all_soft_labels,
        'hard_labels': all_hard_labels
    }

soft_label_data = create_soft_label_dataset(
    TEMPLATE_KEY_TO_USE, 
    TEMPLATE_OPTIONS, 
    NOMINALIZATION_DEMOS
)

if soft_label_data:
    save_filename = f"ft_dataset\{TEMPLATE_KEY_TO_USE}_dataset.pt"
    torch.save(soft_label_data, save_filename)
    print(f"\n Saved to '{save_filename}'.")