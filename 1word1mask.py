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
        #"They are a <mask> because they {w} something.",
        #"They are an <mask> because they {w} something.",
        #"They are the <mask> because they {w} something.",
        # "People who {w} something is identified as a <mask>."
        # "Person who do the action to {w} something is identified as a <mask>."
        # "The person who performs the action of to {w} is identified as a <mask>."
        # "A <mask> is a person who {w} something.",
        # "An <mask> is a person who {w} something.",
        # "The <mask> is a person who {w} something."
        # "The person is called as a <mask> because the main thing that they do is to {W} something.",
        # "The person is called as an <mask> because the main thing that they do is to {W} something.",
        # "The person is called as the <mask> because the main thing that they do is to {W} something."
        # "The main thing that I do is to {w} something, so I am called as <mask>.",

        "I am called as a <mask> because the main thing that I do is to {w} something.",
        "I am called as an <mask> because the main thing that I do is to {w} something.",
        "I am called as the <mask> because the main thing that I do is to {w} something."
    ],
    "agent_vtran":[
        "He is called as a <mask> because the main thing that he does is to {w} something.",
        "He is called as an <mask> because the main thing that he does is to {w} something.",
        "He is called as the <mask> because the main thing that he does is to {w} something.",
        "She is called as a <mask> because the main thing that she does is to {w} something.",
        "She is called as an <mask> because the main thing that she does is to {w} something.",
        "She is called as the <mask> because the main thing that she does is to {w} something."
    ],
    "item_vintran":[
        "The item is called as a <mask> because the main thing that it does is to {w} something.",
        "The item is called as an <mask> because the main thing that it does is to {w} something.",
        "The item is called as the <mask> because the main thing that it does is to {w} something.",
    ],
    "item_vtran":[
        "The item is called as a <mask> because the main thing that it does is to {w}.",
        "The item is called as an <mask> because the main thing that it does is to {w}.",
        "The item is called as the <mask> because the main thing that it does is to {w}.",
    ],
    # Eventuality, TODO
    "evt_vintran": [
        "A <mask> is an eventuality that involves {w} something.",
        "An <mask> is an eventuality that involves {w} something.",
        "Some <mask> is an eventuality that involves {w} something."
    ],
    # TODO
    "state_vintran": [
        "<mask> is the state of to {w} something."
    ],
    # Instrument
    "inst_vintran": [
        "The item is called as a <mask> because the function of it is to {w} something.",
        "The item is called as an <mask> because the main thing that it does is to {w} something.",
        "The item is called as the <mask> because the main thing that it does is to {w} something."
    ],
    "inst_vtran": [
        # "The item is called as a <mask> because it is the instrument to {w}.",
        "The item is called as a <mask> because the main thing that it does is to {w} something.",
        "The item is called as an <mask> because the main thing that it does is to {w} something.",
        "The item is called as the <mask> because the main thing that it does is to {w} something."
        # "A <mask> is needed to {w}",
        # "The <mask> is needed to {w}."
    ],
    #result
    "res_vintran": [
        # "When people {w}, the item that they create is called as <mask>."
        #"The thing that people {w} is <mask>."
        # "<mask> is created because people {w}."
        # "The <mask> is the output result if people {w}."
        "The item is called as a <mask> because it is the result or the act to {w} something.",
        "The item is called as an <mask> because it is the result or the act to {w} something.",
        "The item is called as the <mask> because it is the result or the act to {w} something."
    ],
    "res_vtran": [
        # "When people {w}, the item that they create is called as <mask>."
        #"The thing that people {w} is <mask>."
        # "<mask> is created because people {w}."
        # "The <mask> is the output result if people {w}."
        "The item is called as a <mask> because it is the result or the act to {w}.",
        "The item is called as an <mask> because it is the result or the act to {w}.",
        "The item is called as the <mask> because it is the result or the act to {w}."
    ]
}

# Load pretrained RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
model.eval()  # inference mode

argparser = argparse.ArgumentParser()

# Positional input file and template option so you can run: python 1word1mask.py words.txt eventuality > out.txt
argparser.add_argument("input", help="Input file with one source word per line")
argparser.add_argument("template_option", choices=list(TEMPLATE_OPTIONS.keys()),
                       help="Choose a template option (e.g. nominalization, eventuality)")

argparser.add_argument("-o", "--output", default=None,
                       help="Optional output file (defaults to stdout)")

argparser.add_argument("-n", "--num", type=int, default=10,
                       help="Number of top noun candidates to output per word; <=0 means all vocab")

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

# Use chosen template option (no need to pass templates on command line)
templates = TEMPLATE_OPTIONS[args.template_option]

words = read_lines_guess_encoding(args.input)
out_lines = []

for w in words:
    candidates = predict_candidates(w, templates, args.num, args.scores)
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