from transformers import RobertaTokenizer, RobertaForMaskedLM
import argparse, torch

# Template options
TEMPLATE_OPTIONS = {
    # Nominalization
    "nom_vintran": [
        "A <mask> is defined as when some entities {w} something.",
        "An <mask> is defined as when some entities {w} something.",
        "Some <mask> is defined as when some entities {w} something."
    ],
    "nom_vtran": [
        "A <mask> is defined as when some entities {w}.",
        "An <mask> is defined as when some entities {w}.",
        "The <mask> is defined as when some entities {w}."
    ],
    "agent_vintran":[
        "The main thing that I do is to {w} something, so I am called as <mask>.",
    ],
    "agent_vtran":[
        "They are a <mask> because they {w}.",
        "They are an <mask> because they {w}.",
        "They are the <mask> because they {w}."
    ],
    # Eventuality
    "evt_vintran": [
        "A <mask> is an eventuality that involves {w} something.",
        "An <mask> is an eventuality that involves {w} something.",
        "Some <mask> is an eventuality that involves {w} something."
    ]
}

# Load pretrained RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
model.eval()  # inference mode

argparser = argparse.ArgumentParser("Predict noun forms for a list of words using RoBERTa.")

# Only need input file now - all templates will be processed
argparser.add_argument("input", help="Input file with one source word per line")

argparser.add_argument("-o", "--output", default=None,
                       help="Optional output file (defaults to stdout)")

argparser.add_argument("-n", "--num", type=int, default=5,
                       help="Number of top noun candidates to output per word per template; <=0 means all vocab")

argparser.add_argument("-s", "--scores", action="store_true", default=False,
                       help="Include logit scores in output")

args = argparser.parse_args()

def predict_candidates(src_word, templates, top_k, show_scores=False, filter_alpha=True):
    agg_scores = {}
    for t in templates:
        sent = t.replace("{w}", src_word).replace("<mask>", tokenizer.mask_token)
        model_input = tokenizer(sent, return_tensors="pt")
        mask_positions = (model_input.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            continue
        mask_idx = mask_positions.item()
        with torch.no_grad():
            logits = model(**model_input).logits
        mask_logits = logits[0, mask_idx]
        vocab_size = mask_logits.size(0)
        k = vocab_size if top_k <= 0 else min(top_k, vocab_size)
        topk = torch.topk(mask_logits, k=k, dim=0)
        for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
            token = tokenizer.decode(idx).strip()
            if filter_alpha and not token.isalpha():
                continue
            # optional heuristic: require same initial letter as source word
            if not token or not src_word:
                continue
            if not token[0].lower() == src_word[0].lower():
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

def format_candidates(candidates, num_candidates, show_scores):
    if num_candidates > 0:
        candidates = candidates[:num_candidates]
    if not candidates:
        return ""
    if show_scores:
        return ", ".join([f"{tok}({score:.4f})" for tok, score in candidates])
    else:
        return ", ".join([tok for tok, _ in candidates])

words = read_lines_guess_encoding(args.input)
out_lines = []

# Header row
header = "input_word\tnom_vintran\tnom_vtran\tagent_vintran\tagent_vtran"
out_lines.append(header)

# Process each word with all template options
template_keys = ["nom_vintran", "nom_vtran", "agent_vintran", "agent_vtran"]

for w in words:
    row_data = [w]  # Start with input word
    
    for template_key in template_keys:
        templates = TEMPLATE_OPTIONS[template_key]
        candidates = predict_candidates(w, templates, args.num, args.scores)
        formatted_candidates = format_candidates(candidates, args.num, args.scores)
        row_data.append(formatted_candidates)
    
    out_lines.append("\t".join(row_data))

# Output results
if args.output:
    with open(args.output, "w", encoding="utf-8") as fo:
        fo.write("\n".join(out_lines))
else:
    print("\n".join(out_lines))