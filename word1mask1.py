from transformers import (
    RobertaTokenizer, 
    RobertaForMaskedLM,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import argparse, torch, random
from typing import List, Tuple, Dict
import csv 
import os 
from nom_prompts import TEMPLATE_OPTIONS
from collections import Counter
import string

TSV_FILE_PATH = "nominalization_pairs.tsv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device: {device}")

argparser = argparse.ArgumentParser()

argparser.add_argument("input", help="Input file with one source word per line")
argparser.add_argument("template_option", choices=list(TEMPLATE_OPTIONS.keys()),
                       help="Choose a template option (e.g. nom_vintran)")

argparser.add_argument("--model", type=str, choices=['roberta', 't5'], required=True,
                       help="The model architecture to use.")

argparser.add_argument("--ft", action="store_true", default=False,
                       help="If set, loads the fine-tuned version of the model.")

argparser.add_argument("-o", "--output", default=None,
                       help="Optional output file (defaults to stdout)")

argparser.add_argument("-s", "--scores", action="store_true", default=False,
                       help="Include logit scores in output")

argparser.add_argument("-k", "--k_shot", type=int, default=0,
                       help="Number of few-shot examples (K) to prepend to the prompt.")

argparser.add_argument("-n", "--num", type=int, default=10,
                       help="Number of top noun candidates to output per word; <=0 means all vocab")

args = argparser.parse_args()

model_name = 'roberta-base' if args.model == 'roberta' else 't5-base'

def construct_model_path(args) -> str:
    model_path = model_name 

    if args.ft:
        template_key = args.template_option
        
        ft_path = f"{args.template_option}_{model_name}_finetuned"
        
        if os.path.isdir(ft_path):
             model_path = ft_path
             print(f"INFO: Loading fine-tuned model from: {model_path}")
        else:
             print(f"WARNING: Fine-tuned model directory not found at '{ft_path}'. Falling back to baseline: {model_path}")
             
    return model_path

def load_model(model_name_or_path: str, model_type: str, device: torch.device):
    if model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path) # <--- FIX
        model = RobertaForMaskedLM.from_pretrained(model_name_or_path)
    elif model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path) # <--- FIX
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    model.to(device)
    model.eval()
    
    print(f"{model_type} model successfully loaded: {model_name_or_path}")
    return tokenizer, model

MODEL_PATH_TO_USE = construct_model_path(args)
tokenizer, model = load_model(MODEL_PATH_TO_USE, args.model, device)

def load_and_reorder_pairs(input_filename: str) -> List[Tuple[str, str]]:
    reordered_pairs = []
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) == 2:
                    nominalization = row[0].strip()
                    verb = row[1].strip()
                    reordered_pairs.append((verb, nominalization))
        print(f"Loaded {len(reordered_pairs)} pairs from {input_filename}.")
        return reordered_pairs
    except FileNotFoundError:
        print(f"WARNING: The file '{input_filename}' was not found. Few-shot examples will be empty.")
        return []
    except Exception as e:
        print(f"WARNING: An unexpected error occurred while loading TSV: {e}. Few-shot examples will be empty.")
        return []

loaded_pairs = load_and_reorder_pairs(TSV_FILE_PATH)
NOMINALIZATION_DEMOS = dict(loaded_pairs)

def create_few_shot_prompt(src_word, templates, k_shot, choice):
    available_demos = [(v, n) for v, n in NOMINALIZATION_DEMOS.items() if v != src_word]
    
    if k_shot > 0 and len(available_demos) > 0:
        selected_demos = random.sample(available_demos, min(k_shot, len(available_demos)))
    else:
        selected_demos = []

    full_prompt_components = []
    
    demo_template = templates[0] 
    
    for verb, noun in selected_demos:
        demo_prompt = demo_template.replace("{w}", verb).replace(tokenizer.mask_token, noun)
        full_prompt_components.append(demo_prompt)
        
    query_template = templates[0]
    final_query = query_template.replace("{w}", src_word).replace("<mask>", tokenizer.mask_token)
    
    full_prompt_components.append(final_query)
    
    full_prompt = " ".join(full_prompt_components)
    
    return full_prompt

def predict_candidates_roberta(src_word, templates, top_k, k_shot, show_scores=False):
    
    k_to_use = 0 if args.ft else k_shot

    demo_prompts = []
    if k_to_use > 0:
        available_demos = [(v, n) for v, n in NOMINALIZATION_DEMOS.items() if v != src_word]
        if len(available_demos) > 0:
            selected_demos = random.sample(available_demos, min(k_to_use, len(available_demos)))
            
            demo_template = templates[0] 
            for verb, noun in selected_demos:
                demo_prompt = demo_template.replace("{w}", verb).replace(tokenizer.mask_token, noun)
                demo_prompts.append(demo_prompt)
    
    demo_string = " ".join(demo_prompts)
    
    agg_scores = {}
    
    for query_template in templates:
        
        final_query = query_template.replace("{w}", src_word).replace("<mask>", tokenizer.mask_token)
        
        full_prompt_components = demo_prompts + [final_query]
        sent = " ".join(full_prompt_components)
        
        model_input = tokenizer(sent, return_tensors="pt").to(device)
        
        mask_positions = (model_input.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if mask_positions.numel() == 0:
            print(f"WARNING: No mask found in prompt: {sent}")
            continue
        
        mask_idx = mask_positions[-1].item() 

        with torch.no_grad():
            logits = model(**model_input).logits
        
        mask_logits = logits[0, mask_idx]
        mask_probs = torch.softmax(mask_logits, dim=0) 
        
        for idx in range(mask_probs.size(0)):
            token = tokenizer.decode(idx).strip().lower()
            
            if not token.isalpha() or len(token) < 2:
                continue
            if not token or not src_word or token[0] != src_word[0].lower():
                continue
                
            score = mask_probs[idx].item()
            agg_scores[token] = agg_scores.get(token, 0.0) + score 

    sorted_items = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items

def predict_candidates_t5(src_word, templates, top_k, k_shot, show_scores=False):
    
    k_to_use = 0 if args.ft else k_shot
    demo_prompts = []
    
    if k_to_use > 0:
        available_demos = [(v, n) for v, n in NOMINALIZATION_DEMOS.items() if v != src_word]
        if len(available_demos) > 0:
            selected_demos = random.sample(available_demos, min(k_to_use, len(available_demos)))
            demo_template = templates[0] 
            for verb, noun in selected_demos:
                demo_prompt = demo_template.replace("{w}", verb).replace("<mask>", noun)
                demo_prompts.append(demo_prompt)
    
    all_generated_candidates = []
    num_beams = max(10, top_k) 
    
    for query_template in templates:
        final_query = query_template.replace("{w}", src_word).replace("<mask>", "<extra_id_0>")
        
        full_prompt_components = demo_prompts + [final_query]
        sent = " ".join(full_prompt_components)
        
        model_input = tokenizer(sent, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **model_input,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=10 
            )
        
        candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        translator = str.maketrans('', '', string.punctuation)

        for cand in candidates:
            # 1. Clean punctuation and extra whitespace
            cleaned_cand = cand.strip().lower().translate(translator)
            
            # 2. Split into a list of words
            words = cleaned_cand.split()

            # 3. Check if the list is NOT empty
            if words: 
                # Take the last word
                final_word = words[-1]
                
                # 4. Now apply the filter
                if final_word.isalpha() and len(final_word) > 1:
                    all_generated_candidates.append(final_word)

    candidate_counts = Counter(all_generated_candidates)
    
    sorted_items = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items


def read_lines_guess_encoding(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            return [line.strip() for line in text.splitlines() if line.strip()]
        except Exception:
            continue
    text = raw.decode("latin-1", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]

templates = TEMPLATE_OPTIONS[args.template_option]
words = read_lines_guess_encoding(args.input)
out_lines = []

for w in words:
    
    if args.model == 'roberta':
        candidates = predict_candidates_roberta(w, templates, args.num, args.k_shot, show_scores=args.scores)
    else: # t5
        candidates = predict_candidates_t5(w, templates, args.num, args.k_shot, show_scores=args.scores)

    if args.num > 0:
        candidates = candidates[:args.num]
        
    if args.scores:
        score_label = "score" if args.model == 'roberta' else "freq"
        cand_str = ", ".join([f"{tok}({score_label}:{score:.4f})" for tok, score in candidates])
    else:
        cand_str = ", ".join([tok for tok, _ in candidates])
        
    out_lines.append(f"{w}\t{cand_str}")

if args.output:
    with open(args.output, "w", encoding="utf-8") as fo:
        fo.write("\n".join(out_lines))
else:
    print("\n".join(out_lines))