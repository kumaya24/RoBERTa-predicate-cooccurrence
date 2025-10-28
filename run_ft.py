import torch
import os
import argparse
from transformers import RobertaForMaskedLM, TrainingArguments, Trainer
from nom_prompts import TEMPLATE_OPTIONS # DataCollatorForLanguageModeling is not needed now

# --- Custom Collation Function ---
def tensor_dataset_collator(features):
    """
    Collation function to convert the list of tuples (from TensorDataset) 
    into a dictionary of stacked tensors expected by the Trainer.
    
    Expected feature format: (input_ids, attention_mask, labels)
    """
    # features is a list of tuples: [(ids, mask, labels), (ids, mask, labels), ...]
    
    # Unpack the list of tuples into separate lists of tensors
    input_ids_list = [f[0] for f in features]
    attention_mask_list = [f[1] for f in features]
    labels_list = [f[2] for f in features]
    
    # Stack the lists of tensors to create a single batch tensor
    batch = {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'labels': torch.stack(labels_list),
    }
    return batch

# ---------------------------------


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ... (Argument parsing section remains the same) ...

parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on a custom MLM dataset.")

parser.add_argument(
    "template_option",
    choices=list(TEMPLATE_OPTIONS.keys()),
    help="Choose the template key used to generate the dataset (e.g., nom_vintran)."
)
parser.add_argument(
    "--model",
    type=str,
    choices=['roberta-base'],
    default='roberta-base',
    help="Model we are choosing to use (currently only supports roberta-base)."
)
parser.add_argument(
    "--epochs", 
    type=int, 
    default=20, 
    help="Number of training epochs."
)
parser.add_argument(
    "--batch_size", 
    type=int, 
    default=16, 
    help="Training batch size per device."
)
parser.add_argument(
    "-n", "--num",
    type=int, 
    default=1004,
    help="Number of top noun candidates to output per word; <=0 means all vocab."
)

args = parser.parse_args()

TEMPLATE_KEY_TO_USE = args.template_option 
MODEL_NAME = args.model 
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

DATASET_FILE_PATH = f"{TEMPLATE_KEY_TO_USE}_mlm_finetune_dataset.pt"
OUTPUT_DIR = f"roberta_{TEMPLATE_KEY_TO_USE}_{MODEL_NAME}_finetuned"

print(f"Loading dataset from: {DATASET_FILE_PATH}")
try:
    # Use the fix from our previous discussion: weights_only=False
    train_dataset = torch.load(DATASET_FILE_PATH, weights_only=False)
    
    model = RobertaForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)

except FileNotFoundError:
    print(f"ERROR: Dataset file not found at {DATASET_FILE_PATH}. Please run pre-finetuning.py first.")
    exit()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=500, 
    logging_steps=100,
    learning_rate=5e-5,
    do_train=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # --- FIX APPLIED HERE ---
    data_collator=tensor_dataset_collator 
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"\n Fine-tuning complete. Model saved to '{OUTPUT_DIR}'")