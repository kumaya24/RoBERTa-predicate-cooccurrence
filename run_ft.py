import torch
import os
import argparse
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, TrainingArguments, Trainer
from torch.utils.data import Dataset
from typing import List, Dict
import torch.utils.data.dataloader

from nom_prompts import TEMPLATE_OPTIONS 

# --- Custom Dataset Class ---
class SoftLabelDataset(Dataset):
    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        self.input_ids = data_dict['input_ids']
        self.attention_mask = data_dict['attention_mask']
        self.soft_labels = data_dict['soft_labels']
        self.hard_labels = data_dict['hard_labels'] 

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx: int):
        # We ensure the tensor type is consistent and return the dict
        return {
            'input_ids': self.input_ids[idx].clone().detach(),
            'attention_mask': self.attention_mask[idx].clone().detach(),
            'labels': self.hard_labels[idx].clone().detach(),  
            'soft_labels': self.soft_labels[idx].clone().detach() 
        }

# --- Custom Collation Function (Simplified and Robust) ---
def soft_label_data_collator(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collator that properly handles soft_labels alongside standard fields."""
    
    # Extract soft_labels separately
    soft_labels_list = [f['soft_labels'] for f in features]
    
    # Create copies of features without soft_labels for default collation
    features_without_soft = [{k: v for k, v in f.items() if k != 'soft_labels'} for f in features]
    
    # Use default collation for standard fields
    batch = torch.utils.data.dataloader.default_collate(features_without_soft)
    
    # Add soft_labels back to the batch
    batch['soft_labels'] = torch.stack(soft_labels_list)
    
    return batch


# --- Custom Trainer Class with KL Divergence Loss (Key Fixes) ---
class KLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # ACCESS soft_labels directly and REMOVE from inputs later if needed
        soft_labels = inputs.get("soft_labels")
        
        if soft_labels is None:
             # Should not happen if collator is working, but safety fallback
             raise KeyError(f"Soft labels were not found. Available keys: {list(inputs.keys())}")
        
        # Remove soft_labels before passing to the base model's forward method
        inputs_for_model = {k: v for k, v in inputs.items() if k != "soft_labels"}
        
        # 2. Forward pass through the student model
        outputs = model(**inputs_for_model)
        logits = outputs.logits
        
        # 3. Find the mask position
        mask_positions = (inputs_for_model['input_ids'][0] == model.config.mask_token_id).nonzero(as_tuple=True)[0]
        
        if mask_positions.numel() == 0:
            loss = super().compute_loss(model, inputs_for_model) # Use inputs_for_model here
            return (loss, outputs) if return_outputs else loss
            
        mask_idx = mask_positions[0].item()

        # 4. Extract Logits at the Mask Position
        student_logits = logits[:, mask_idx, :] 
        teacher_logits = soft_labels.to(student_logits.device)
        
        # 5. Calculate KL Divergence Loss
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


# --- Configuration and Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Fine-tune RoBERTa using Soft Labels (KL Divergence).")

parser.add_argument("template_option", choices=list(TEMPLATE_OPTIONS.keys()),
                       help="Choose the template key used to generate the dataset (e.g., nom_vintran).")
parser.add_argument("--model", type=str, choices=['roberta-base'], default='roberta-base',
                       help="Model we are choosing to use (roberta-base).")
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size per device.")

args = parser.parse_args()

TEMPLATE_KEY_TO_USE = args.template_option 
MODEL_NAME = args.model 

# --- Paths ---
DATASET_FILE_PATH = f"{TEMPLATE_KEY_TO_USE}_soft_label_dataset.pt" 
OUTPUT_DIR = f"roberta_{TEMPLATE_KEY_TO_USE}_{MODEL_NAME}_soft_finetuned"

# --- Data Loading and Preparation ---

print(f"Loading soft-label data from: {DATASET_FILE_PATH}")
print(f"Using device: {DEVICE}")

try:
    soft_data_dict = torch.load(DATASET_FILE_PATH, weights_only=False)
    
    train_dataset = SoftLabelDataset(soft_data_dict)
    
    model = RobertaForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.config.mask_token_id = 50264 

    print(f"INFO: Dataset size: {len(train_dataset)}")
    print(f"INFO: Model loaded on {DEVICE}")

except FileNotFoundError:
    print(f"ERROR: Soft-label dataset file not found at {DATASET_FILE_PATH}. Please run the revised pre-finetuning.py first.")
    exit()

# --- Define Training Arguments ---

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    do_train=True,
    report_to="none",
    weight_decay=0.0,
    remove_unused_columns=False,  # CRITICAL: Don't remove soft_labels!
    use_cpu=False,  # Use GPU if available
    no_cuda=False,  # Enable CUDA
)

# --- Initialize and Run the Custom Trainer ---

trainer = KLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=soft_label_data_collator 
)

print("\nStarting soft-label fine-tuning (Knowledge Distillation)...")
trainer.train()

# --- Save Final Model ---
trainer.save_model(OUTPUT_DIR)
print(f"\n✅ Fine-tuning complete. Model saved to '{OUTPUT_DIR}'.")