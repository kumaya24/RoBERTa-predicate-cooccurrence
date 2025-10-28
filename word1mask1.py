from transformers import RobertaTokenizer, RobertaForMaskedLM
import argparse, torch, random
from typing import List, Tuple, Dict
import csv 
import os 
from nom_prompts import TEMPLATE_OPTIONS

TSV_FILE_PATH = "nominalization_pairs.tsv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device: {device}")

argparser = argparse.ArgumentParser()

argparser.add_argument("input", help="Input file with one source word per line")
argparser.add_argument("template_option", choices=list(TEMPLATE_OPTIONS.keys()),
                       help="Choose a template option (e.g. nom_vintran)")

argparser.add_argument("--model", type=str, default="roberta-base",
                       choices=["roberta-base"],
                       help="The name of the baseline model (e.g., roberta-base).")

argparser.add_argument("--ft", action="store_true", default=False,
                       help="If set, loads the fine-tuned version of the model.")

argparser.add_argument("-o", "--output", default=None,
                       help="Optional output file (defaults to stdout)")

argparser.add_argument("-s", "--scores", action="store_true", default=False,
                       help="Include logit scores in output")

argparser.add_argument("-k", "--k_shot", type=int, default=10,
                       help="Number of few-shot examples (K) to prepend to the prompt.")

argparser.add_argument("-n", "--num", type=int, default=10,
                       help="Number of top noun candidates to output per word; <=0 means all vocab")

args = argparser.parse_args()


def construct_model_path(args) -> str:
    model_path = args.model 

    if args.ft:
        template_key = args.template_option
        baseline_name = args.model
        
        # NOTE: This string must EXACTLY match the output directory of run_ft.py
        ft_path = f"{baseline_name.split('-')[0]}_{template_key}_{baseline_name}_finetuned"
        
        if os.path.isdir(ft_path):
             model_path = ft_path
             print(f"INFO: Loading fine-tuned model from: {model_path}")
        else:
             print(f"WARNING: Fine-tuned model directory not found at '{ft_path}'. Falling back to baseline: {model_path}")
             
    return model_path

def load_mlm_model(model_name_or_path: str, baseline_name: str, device: torch.device):
    tokenizer = RobertaTokenizer.from_pretrained(baseline_name)
    
    model = RobertaForMaskedLM.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    
    print(f"INFO: Model successfully loaded: {model_name_or_path}")
    return tokenizer, model

MODEL_PATH_TO_USE = construct_model_path(args)
tokenizer, model = load_mlm_model(MODEL_PATH_TO_USE, args.model, device)

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
        print(f"INFO: Successfully loaded {len(reordered_pairs)} pairs from {input_filename}.")
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

def predict_candidates(src_word, templates, top_k, choice, k_shot, show_scores=False):
    agg_scores = {}
    
    full_few_shot_prompt = create_few_shot_prompt(src_word, templates, k_shot, choice)
    
    sent = full_few_shot_prompt 
    model_input = tokenizer(sent, return_tensors="pt").to(device)
    
    mask_positions = (model_input.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if mask_positions.numel() == 0:
        return []
    
    mask_idx = mask_positions[-1].item() 

    with torch.no_grad():
        logits = model(**model_input).logits
    
    mask_logits = logits[0, mask_idx]
    vocab_size = mask_logits.size(0)
    k = vocab_size if top_k <= 0 else min(top_k * 5, vocab_size)
    
    topk = torch.topk(mask_logits, k=k, dim=0)

    for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
        token = tokenizer.decode(idx).strip().lower()
        
        if not token.isalpha():
            continue

        if not token or not src_word or token[0] != src_word[0].lower():
            continue

        prev = agg_scores.get(token)
        if prev is None or val > prev:
            agg_scores[token] = val

    sorted_items = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)
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
    candidates = predict_candidates(w, templates, args.num, args.template_option, args.k_shot, show_scores=args.scores)
    if args.num > 0:
        candidates = candidates[:args.num]
    if args.scores:
        cand_str = ", ".join([f"{tok}({score:.4f})" for tok, score in candidates])
    else:
        cand_str = ", ".join([tok for tok, _ in candidates])
    out_lines.append(f"{w}\t{cand_str}")

if args.output:
    with open(args.output, "w", encoding="utf-8") as fo:
        fo.write("\n".join(out_lines))
else:
    print("\n".join(out_lines))