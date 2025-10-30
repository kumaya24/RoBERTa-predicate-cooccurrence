import torch
import os
import argparse
import torch.nn.functional as F
from transformers import (
    TrainingArguments, 
    Trainer,
    RobertaForMaskedLM, 
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset
from typing import List, Dict
import torch.utils.data.dataloader
import csv

from nom_prompts import TEMPLATE_OPTIONS 

class SoftLabelDataset(Dataset):
    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        self.input_ids = data_dict['input_ids']
        self.attention_mask = data_dict['attention_mask']
        self.soft_labels = data_dict['soft_labels']
        self.hard_labels = data_dict['hard_labels'] 

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx: int):
        return {
            'input_ids': self.input_ids[idx].clone().detach(),
            'attention_mask': self.attention_mask[idx].clone().detach(),
            'labels': self.hard_labels[idx].clone().detach(), 
            'soft_labels': self.soft_labels[idx].clone().detach() 
        }

def soft_label_data_collator(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    soft_labels_list = [f['soft_labels'] for f in features]
    
    features_without_soft = [{k: v for k, v in f.items() if k != 'soft_labels'} for f in features]
    
    batch = torch.utils.data.dataloader.default_collate(features_without_soft)
    
    batch['soft_labels'] = torch.stack(soft_labels_list)
    
    return batch


class KLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        soft_labels = inputs.get("soft_labels")
        
        if soft_labels is None:
            raise KeyError(f"Soft labels were not found. Available keys: {list(inputs.keys())}")
        
        inputs_for_model = {k: v for k, v in inputs.items() if k != "soft_labels"}
        
        outputs = model(**inputs_for_model)
        logits = outputs.logits
        
        mask_positions = (inputs_for_model['input_ids'][0] == model.config.mask_token_id).nonzero(as_tuple=True)[0]
        
        if mask_positions.numel() == 0:
            loss = super().compute_loss(model, inputs_for_model)
            return (loss, outputs) if return_outputs else loss
            
        mask_idx = mask_positions[0].item()

        student_logits = logits[:, mask_idx, :] 
        teacher_logits = soft_labels.to(student_logits.device)
        
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        kl_loss = F.kl_div(
            input=student_log_probs, 
            target=teacher_probs, 
            reduction="batchmean",
            log_target=False
        )
        
        total_loss = kl_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
# T5 Dataset 
class Seq2SeqDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data 
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        
        # Tokenize Input
        tokenized_input = self.tokenizer(
            item['input_text'],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize Target (Labels)
        tokenized_target = self.tokenizer(
            item['target_text'],
            max_length=self.max_len, 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized_input['input_ids'].squeeze(0),
            'attention_mask': tokenized_input['attention_mask'].squeeze(0),
            'labels': tokenized_target['input_ids'].squeeze(0)
        }

# T5 Data Loading Helper 
def load_and_reorder_pairs(input_filename: str) -> Dict[str, str]:
    """Loads TSV file and returns {verb: noun} mapping."""
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
TSV_FILE_PATH = "nominalization_pairs.tsv" # For T5

parser = argparse.ArgumentParser(description="Fine-tune RoBERTa using Soft Labels (KL Divergence).")

parser.add_argument("template_option", choices=list(TEMPLATE_OPTIONS.keys()),
                    help="Choose the template key used to generate the dataset (e.g., nom_vintran).")
parser.add_argument("--model", type=str, choices=['roberta', 't5'], required=True,
                    help="Model we are choosing to use (roberta-base).")
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size per device.")

args = parser.parse_args()

TEMPLATE_KEY_TO_USE = args.template_option 
if args.model == 'roberta':
    MODEL_NAME = 'roberta-base'
else:
    MODEL_NAME = 't5-base'

DATASET_FILE_PATH = f"ft_dataset/{TEMPLATE_KEY_TO_USE}_dataset.pt"
OUTPUT_DIR = f"{TEMPLATE_KEY_TO_USE}_{MODEL_NAME}_finetuned"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=500,
    logging_steps=100,
    learning_rate=1e-5, # Kept the 1e-5 learning rate
    do_train=True,
    report_to="none",
    weight_decay=0.0,
    use_cpu=False,
    no_cuda=False,
)

if args.model == 'roberta':
    # Roberta
    
    DATASET_FILE_PATH = f"ft_dataset/{TEMPLATE_KEY_TO_USE}_dataset.pt" 
    print(f"Loading RoBERTa data from: {DATASET_FILE_PATH}")
    
    try:
        soft_data_dict = torch.load(DATASET_FILE_PATH, weights_only=False)
        train_dataset = SoftLabelDataset(soft_data_dict)
        model = RobertaForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
        model.config.mask_token_id = 50264 
        print(f"INFO: RoBERTa Model loaded on {DEVICE}")

        # --- THIS WAS THE MISSING LINE ---
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME) 

    except FileNotFoundError:
        print(f"ERROR: Soft-label dataset file not found at {DATASET_FILE_PATH}.")
        print("Please run pre-finetuning.py first for the roberta model.")
        exit()

    training_args.remove_unused_columns = False 

    trainer = KLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=soft_label_data_collator,
        tokenizer=tokenizer
    )

else: 
    # T5
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    print(f"Loading raw pairs from: {TSV_FILE_PATH}")
    nominalization_pairs = load_and_reorder_pairs(TSV_FILE_PATH)
    templates = TEMPLATE_OPTIONS.get(TEMPLATE_KEY_TO_USE)
    
    if not templates:
        print(f"ERROR: Template key '{TEMPLATE_KEY_TO_USE}' not found in nom_prompts.py")
        exit()

    # Create the T5-formatted dataset
    seq2seq_data = []
    for verb, target_noun in nominalization_pairs.items():
        for template in templates:
            input_text = template.replace("{w}", verb).replace("<mask>", "<extra_id_0>")
            
            target_text = target_noun
            
            seq2seq_data.append({
                'input_text': input_text,
                'target_text': target_text
            })
            
    train_dataset = Seq2SeqDataset(seq2seq_data, tokenizer)
    
    # Seq2Seq collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"\n Model saved to '{OUTPUT_DIR}'.")