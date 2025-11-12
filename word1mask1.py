from transformers import (
    RobertaTokenizer, 
    RobertaForMaskedLM,
    T5TokenizerFast,  # Use Fast
    T5ForConditionalGeneration
)
import argparse, torch, random
from typing import List, Tuple, Dict
import csv 
import os 
from nom_prompts import TEMPLATE_OPTIONS
from collections import Counter
import string
import nltk  # Import NLTK module
from nltk.corpus import wordnet as wn  # Import WordNet
import jellyfish
from os.path import commonprefix


AGENT_SYNSETS = set()

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

argparser.add_argument("-n", "--num", type=int, default=20,
                       help="Number of top noun candidates to output per word; <=0 means all vocab")


def get_semantic_score(word: str, is_agent_task: bool) -> int:
    noun_synsets = wn.synsets(word, pos=wn.NOUN)

    if not noun_synsets:
        return 0 

    if is_agent_task:
        for syn in noun_synsets:
            all_hypernyms = set(syn.closure(lambda s: s.hypernyms()))
            all_hypernyms.add(syn)
            
            if not AGENT_SYNSETS.isdisjoint(all_hypernyms):
                return 1  
        return 0
    else:
        return 1

def construct_model_path(args, model_name_str) -> str:
    model_path = model_name_str
    if args.ft:
        template_key = args.template_option
        ft_path = f"{args.template_option}_{model_name_str}_finetuned"
        if os.path.isdir(ft_path):
             model_path = ft_path
             print(f"INFO: Loading fine-tuned model from: {model_path}")
        else:
             print(f"WARNING: Fine-tuned model directory not found at '{ft_path}'. Falling back to baseline: {model_path}")
    return model_path

def load_model(model_name_or_path: str, model_type: str, device: torch.device):
    if model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
        model = RobertaForMaskedLM.from_pretrained(model_name_or_path)
    elif model_type == 't5':
        tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.to(device)
    model.eval()
    print(f"{model_type} model successfully loaded: {model_name_or_path}")
    return tokenizer, model

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

def predict_candidates_roberta(src_word, templates, top_k, k_shot, show_scores=False, demos={}):
    k_to_use = 0 if args.ft else k_shot
    demo_prompts = []
    if k_to_use > 0:
        available_demos = [(v, n) for v, n in demos.items() if v != src_word]
        if len(available_demos) > 0:
            selected_demos = random.sample(available_demos, min(k_to_use, len(available_demos)))
            demo_template = templates[0] 
            for verb, noun in selected_demos:
                demo_prompt = demo_template.replace("{w}", verb).replace(tokenizer.mask_token, noun)
                demo_prompts.append(demo_prompt)
    
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


def predict_candidates_t5(src_word, templates, top_k, k_shot, show_scores=False, is_agent_task_flag=False, demos={}):
    
    k_to_use = 0 if (args.ft or not demos) else k_shot
    demo_prompts = []
    
    if k_to_use > 0:
        available_demos = [(v, n) for v, n in demos.items() if v != src_word]
        if len(available_demos) > 0:
            selected_demos = random.sample(available_demos, min(k_to_use, len(available_demos)))
            demo_template = templates[0] 
            for verb, noun in selected_demos:
                demo_prompt = demo_template.replace("{w}", verb).replace("<mask>", noun)
                demo_prompts.append(demo_prompt)
    
    all_generated_candidates = []
    num_beams = max(20, top_k) 
    
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
            cleaned_cand = cand.strip().lower().translate(translator)
            words = cleaned_cand.split()

            if words: 
                final_word = words[-1]
                if final_word.isalpha() and len(final_word) > 1:
                    all_generated_candidates.append(final_word)

    
    candidate_counts = Counter(all_generated_candidates)
    reranked_candidates = []


    if top_k > 0:
        top_n_by_freq = candidate_counts.most_common(top_k)
    else:
        top_n_by_freq = candidate_counts.most_common()

    for cand, freq in candidate_counts.items():
        
        noun_synsets = wn.synsets(cand, pos=wn.NOUN)
        if not noun_synsets:
            reranked_candidates.append((cand, 0.0))
            continue 

        semantic_score = get_semantic_score(cand, is_agent_task_flag) 

        prefix_score = len(commonprefix([src_word, cand]))
        
        dist = jellyfish.levenshtein_distance(src_word, cand)
        distance_score = -dist 

        if is_agent_task_flag and cand == src_word:
            prefix_score = 0 
        elif not is_agent_task_flag and cand == src_word:
            prefix_score = 10 
            
        final_score = (prefix_score * 1000) + (semantic_score * 500) + distance_score + freq
        
        reranked_candidates.append( (cand, final_score) )
    
    sorted_items = sorted(reranked_candidates, key=lambda x: x[1], reverse=True)
    
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

# --- This is the main part of the script ---
if __name__ == "__main__":
    
    args = argparser.parse_args()

    # --- OPTIMIZATION 1: Conditionally load NLTK & WordNet ---
    IS_AGENT_TASK = args.model == 't5' and args.template_option.startswith("agent_")
    
    # We need WordNet for *all* T5 tasks now, to check if a word is a noun.
    if args.model == 't5':
        print("INFO: T5 task detected, checking NLTK/WordNet...")
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            print("--- NLTK DATA MISSING ---")
            print("First-time setup: Downloading WordNet data. This may take a moment...")
            nltk.download('wordnet')
            print("Download complete.")
        
        # Only populate the AGENT set if it's an agent task
        if IS_AGENT_TASK:
            AGENT_SYNSETS.add(wn.synset('person.n.01'))
            AGENT_SYNSETS.add(wn.synset('agent.n.01'))
            AGENT_SYNSETS.add(wn.synset('causal_agent.n.01'))
            print("INFO: WordNet agent synsets loaded.")
        else:
             print("INFO: WordNet loaded (for general noun-checking).")


    # --- OPTIMIZATION 2: Conditionally load K-shot examples ---
    NOMINALIZATION_DEMOS = {}
    # Only load if k_shot > 0 AND we are not using a fine-tuned model
    if args.k_shot > 0 and not args.ft:
        print(f"INFO: k_shot={args.k_shot}, loading few-shot examples...")
        loaded_pairs = load_and_reorder_pairs(TSV_FILE_PATH)
        NOMINALIZATION_DEMOS = dict(loaded_pairs)
    else:
        print("INFO: Skipping few-shot example loading.")

    # --- Model Loading ---
    model_name_str = 'roberta-base' if args.model == 'roberta' else 'google/flan-t5-base' # <-- CRITICAL: Use flan-t5
    MODEL_PATH_TO_USE = construct_model_path(args, model_name_str)
    
    tokenizer, model = load_model(MODEL_PATH_TO_USE, args.model, device)

    # --- Main Loop ---
    templates = TEMPLATE_OPTIONS[args.template_option]
    words = read_lines_guess_encoding(args.input)
    out_lines = []

    for w in words:
        if args.model == 'roberta':
            candidates = predict_candidates_roberta(w, templates, args.num, args.k_shot, show_scores=args.scores, demos=NOMINALIZATION_DEMOS)
        else: # t5
            candidates = predict_candidates_t5(w, templates, args.num, args.k_shot, show_scores=args.scores, is_agent_task_flag=IS_AGENT_TASK, demos=NOMINALIZATION_DEMOS)

        if args.num > 0:
            candidates = candidates[:args.num]
            
        if args.scores:
            score_label = "score" 
            cand_str = ", ".join([f"{tok}({score_label}:{score:.4f})" for tok, score in candidates])
        else:
            cand_str = ", ".join([tok for tok, _ in candidates])
            
        out_lines.append(f"{w}\t{cand_str}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fo:
            fo.write("\n".join(out_lines))
    else:
        print("\n".join(out_lines))