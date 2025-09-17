from transformers import RobertaTokenizer, RobertaForMaskedLM
import argparse, torch

# Template options
TEMPLATE_OPTIONS = {
    # Nominalization
    "nom_vintran": [
        # "In English, a <mask> is defined as when some entities {w}.",
        # "In English, an <mask> is defined as when some entities {w}.",
        # "In English, some <mask> is defined as when some entities {w}.",
        # "In English, the <mask> is defined as when some entities {w}."
        # "Technically, a <mask> is defined as when some entities {w}.",
        # "Technically, an <mask> is defined as when some entities {w}.",
        # "Technically, some <mask> is defined as when some entities {w}.",
        # "Technically, the <mask> is defined as when some entities {w}."
        # "Literally, a <mask> is defined as when some entities {w}.",
        # "Literally, an <mask> is defined as when some entities {w}.",
        # "Literally, the <mask> is defined as when some entities {w}.",
        # "Literally, some <mask> is defined as when some entities {w}."
        #"A <mask> is exclusively defined as when some entities {w}.",
        #"An <mask> is exclusively defined as when some entities {w}.",
        #"Some <mask> is exclusively defined as when some entities {w}.",
        #"The <mask> is exclusively defined as when some entities {w}."
        # "A <mask> is defined as the state in which some entities {w}.",
        # "An <mask> is defined as the state in which some entities {w}.",
        # "The <mask> is defined as the state in which some entities {w}.",
        # "Some <mask> is defined as the state in which some entities {w}."
        #"A <mask> is defined as a way when someone or something {w}.",
        #"An <mask> is defined as a way when someone or something {w}.",
        #"Some <mask> is defined as a way when someone or something {w}.",
        #"The <mask> is defined as a way when someone or something {w}."
    # NOT BAD: "The action that people {w} is defined as <mask>.",
        # "The meaning of <mask> is for people to {w}.",
        # "<mask> is created for some entities to {w}.",
        # "<mask> is defined as a practice through which people {w}."

        
        #"A <mask> is defined as a way to describe when someone or something {w}.",
        #"An <mask> is defined as a way to describe when someone or something {w}.",
        #"Some <mask> is defined as a way to describe when someone or something {w}.",
        #"The <mask> is defined as a way to describe when someone or something {w}."
        #"A <mask> is defined as when some entities {w}, or as a way to describe when someone or something {w}.",
        #"An <mask> is defined as when some entities {w}, or as a way to describe when someone or something {w}.",
        #"The <mask> is defined as when some entities {w}, or as a way to describe when someone or something {w}.",
        #"Some <mask> is defined as when some entities {w}, or as a way to describe when someone or something {w}."
        
        
        
        # When only use 2nd set of prompts, "technically" makes it worse. 

        # Do -> Deed (good)
        # Tell -> tale (too low)
        # Try -> trial (no)
        # Remember -> remembrance (good)
        # Eat -> event as first place..., eating is first place when only 2nd sets of prompts
        # Starve (bad())

        #"In strict terms, a <mask> is defined as when some entities {w}.",
        #"In strict terms, an <mask> is defined as when some entities {w}.",
        #"In strict terms, the <mask> is defined as when some entities {w}.",
        #"In strict terms, some <mask> is defined as when some entities {w}."
        # X tale, x deed, x trial, x remembrance
        #"{w} is defined as to cause someone to come to a <mask>."

        # -->>>>
        "A <mask> is defined as when some entities {w}.",
        "An <mask> is defined as when some entities {w}.",
        "Some <mask> is defined as when some entities {w}.",
        "The <mask> is defined as when some entities {w}.",
        
        
        "A <mask> is defined as the act of {w}.",
        "An <mask> is defined as the act of {w}.",
        "The <mask> is defined as the act of {w}.",
        "Some <mask> is defined as the act of {w}.",

    ],
    "nom_vtran": [
        "A <mask> is defined as when some entities {w} something.",
        "An <mask> is defined as when some entities {w} something.",
        "The <mask> is defined as when some entities {w} something.",
        "Some <mask> is defined as when some entities {w} something.",
        #"Technically, a <mask> is defined as when some entities {w} something.",
        #"Technically, an <mask> is defined as when some entities {w} something.",
        #"Technically, the <mask> is defined as when some entities {w} something.",
        #"Technically, some <mask> is defined as when some entities {w} something.",
        #"In English, a <mask> is literally defined as when some entities {w} something.",
        #"In English, an <mask> is literally defined as when some entities {w} something.",
        #"In English, the <mask> is literally defined as when some entities {w} something.",
        #"In English, some <mask> is literally defined as when some entities {w} something.",

        # "The thing that people {w} something is defined as <mask>."
        # "The action that people {w} something is defined as <mask>."
        # "<mask> is to describe that people {w} something."

        # Show many og form as well
        "A <mask> is defined as the act of {w} something.",
        "An <mask> is defined as the act of {w} something.",
        "The <mask> is defined as the act of {w} something.",
        "Some <mask> is defined as the act of {w} something.",

       
    ],
    "agent_vintran":[
        #"They are a <mask> because they {w} something.",
        #"They are an <mask> because they {w} something.",
        #"They are the <mask> because they {w} something.",
        # "People who {w} something is identified as a <mask>."
        # "Person who do the action to {w} something is identified as a <mask>."
        # "The person who performs the action of to {w} is identified as a <mask>."
        #"A <mask> is a person who {w} something.",
        #"An <mask> is a person who {w} something.",
        #"The <mask> is a person who {w} something."
        # "The person is called as a <mask> because the main thing that they do is to {W} something.",
        # "The person is called as an <mask> because the main thing that they do is to {W} something.",
        # "The person is called as the <mask> because the main thing that they do is to {W} something."
        # "The main thing that I do is to {w} something, so I am called as <mask>.",
        # "I am called as a <mask> because the main thing that I do is to {w} something.",
        # "I am called as an <mask> because the main thing that I do is to {w} something.",
        # "I am called as the <mask> because the main thing that I do is to {w} something."

        #"Technically, A <mask> is defined as someone or something that will {w}.",
        #"Technically, An <mask> is defined as someone or something that will {w}.",
        #"Technically, The <mask> is defined as someone or something that will {w}.",
        #"Technically, Some <mask> is defined as someone or something that will {w}."

        # "<mask> is defined as the person who {w}."
        # "<mask> is the profession that {w} regularly.",

        #"In English, a <mask> is defined as someone who will {w} regularly.",
        #"In English, an <mask> is defined as someone who will {w} regularly.",
        #"In English, the <mask> is defined as someone who will {w} regularly.",
        # "In English, some <mask> is defined as someone who will {w} regularly."

        "A <mask> refers to a person who {w} regularly.",
        "An <mask> refers to a person who {w} regularly.",
        "The <mask> refers to a person who {w} regularly.", 
        #"Some <mask> refers to a person who {w} regularly."
        #"In English, something or someone able to {w}  is defined as a <mask>.",
        #"In English, something or someone able to {w}  is defined as an <mask>.",
        #"In English, something or someone able to {w}  is defined as the <mask>."
        
        #"If someone or something does {w} frequently, they are therefore a <mask>.",
        #"If someone or something does {w} frequently, they are therefore an <mask>.",
        #"If someone or something does {w} frequently, they are therefore the <mask>."
    ],
    "agent_vtran":[
        # "He is called as a <mask> because the main thing that he does is to {w} something.",
        # "He is called as an <mask> because the main thing that he does is to {w} something.",
        # "He is called as the <mask> because the main thing that he does is to {w} something.",
        # "She is called as a <mask> because the main thing that she does is to {w} something.",
        # "She is called as an <mask> because the main thing that she does is to {w} something.",
        # "She is called as the <mask> because the main thing that she does is to {w} something."

        #"A <mask> is defined as someone that will {w} someone or something.",
        #"An <mask> is defined as someone that will {w} someone or something.",
        # "<mask> is defined as someone or something that will {w} something."

        #"Technically, A <mask> is defined as someone who {w} something regularly.",
        #"Technically, An <mask> is defined as someone who {w} something regularly.",
        #"Technically, The <mask> is defined as someone who {w} something regularly.",
        #"Technically, Some <mask> is defined as someone who {w} something regularly."


        #"A <mask> is defined as someone who {w} something regularly.",
        #"An <mask> is defined as someone who {w} something regularly.",     
        #"The <mask> is defined as someone who {w} something regularly.",
        #"Some <mask> is defined as someone who {w} something regularly.",

        #"A <mask> is defined as someone whose job is to {w} something regularly.",
        #"An <mask> is defined as someone whose job is to {w} something regularly.",
        #"The <mask> is defined as someone whose job is to {w} something regularly.",
        #"Some <mask> is defined as someone whose job is to {w} something regularly."

        "A <mask> is defined as a person whose work is to {w} things.",
        "An <mask> is defined as a person whose work is to {w} things.",
        "The <mask> is defined as a person whose work is to {w} things."
    ],
    
    # Eventuality
    "evt_vintran": [
        # "A <mask> is an eventuality that involves {w}.",
        # "An <mask> is an eventuality that involves {w}.",
        # "Some <mask> is an eventuality that involves {w}."
        # "Technically, <mask> is defined as when someone or something will {w}.",
        # "Technically, a <mask> is defined as when someone or something will {w}.",
        # "Technically, an <mask> is defined as when someone or something will {w}."
        "In English, <mask> is defined as when someone or something will {w}.",
        "In English, a <mask> is defined as when someone or something will {w}.",
        "In English, an <mask> is defined as when someone or something will {w}."
    ],
    "evt_vtran": [
        # "A <mask> is an eventuality that involves {w} something.",
        # "An <mask> is an eventuality that involves {w} something.",
        # "Some <mask> is an eventuality that involves {w} something."

        # "<mask> is defined as when someone or something will {w} someone or something."
        # "A <mask> is defined as when someone or something will {w} someone or something.",
        # "An <mask> is defined as when someone or something will {w} someone or something.",
        #"Some <mask> is defined as when someone or something will {w} someone or something."

        "In English, <mask> is defined as when someone or something will {w} someone or something.",
        "In English, a <mask> is defined as when someone or something will {w} someone or something.",
        "In English, an <mask> is defined as when someone or something will {w} someone or something.",
        "In English, some <mask> is defined as when someone or something will {w} someone or something.",
        "In English, the <mask> is defined as when someone or something will {w} someone or something."
        #"A <mask> is defined as a event of {w} something.",
        #"An <mask> is defined as a event of {w} something.",
        #"The <mask> is defined as a event of {w} something.",
        #"Some <mask> is defined as a event of {w} something."
    ],

    #### V -> ADJ
    "participleAdj_vintran":[
        # "people say 'this is <mask>!' when they {w}."
        "In English, something or someone able to {w} is defined as being <mask>."
    ],
    "participleAdj_vtran":[
        #"people say 'this is <mask>!' when they {w} it.",
        # "Something is very <mask> is defined as when something is capable to be {w}."
        #"The word {w} can be converted to <mask> to describe things."
        # "Very <mask> is defined as when something can {w} something."
        # "In English, the sentence that something can {w} something is as same as the sentence that something is very <mask>."
        # "In English, they can {w} something, which also means that something is very <mask>."
        "In English, something or someone able to {w} something is defined as being <mask>."
    ],

    ##### Causative
    "causative_vintran": [
        # "In English, for someone or something to <mask> someone or something is defined as to cause it to {w}.",
    ],
    "causative_vtran": [
        "In English, for someone or something to {w} someone or something is defined as to cause it to <mask>.",
    ],
    ## DISCARDED
        #result
    "res_vintran": [
        # "When people {w}, the item that they create is called as <mask>."
        #"The thing that people {w} is <mask>."
        # "<mask> is created because people {w}."
        # "The <mask> is the output result if people {w}."
        # "The item is called as a <mask> because it is the result or the act to {w} .",
        # "The item is called as an <mask> because it is the result or the act to {w} .",
        # "The item is called as the <mask> because it is the result or the act to {w} ."
        # "Technically, <mask> is defined as the result when someone or something use it to {w}."
        "In English, <mask> is defined as the result when someone or something {w}."
    ],
    "res_vtran": [
        # "When people {w}, the item that they create is called as <mask>."
        #"The thing that people {w} is <mask>."
        # "<mask> is created because people {w}."
        # "The <mask> is the output result if people {w}."
        "The item is called as a <mask> because it is the result or the act to {w} something.",
        "The item is called as an <mask> because it is the result or the act to {w} something.",
        "The item is called as the <mask> because it is the result or the act to {w} something."
    ],
    "item_vintran":[
        # "The item is called as a <mask> because the main thing that it does is to {w} something.",
        # "The item is called as an <mask> because the main thing that it does is to {w} something.",
        # "The item is called as the <mask> because the main thing that it does is to {w} something.",

        #"A <mask> is defined as something that will {w}.",
        #"An <mask> is defined as something that will {w}."

        # "In English, <mask> is defined as something that will {w}."
        "A <mask> is defined as something that {w} regularly.",
        "An <mask> is defined as something that {w} regularly.",
        "The <mask> is defined as something that {w} regularly.",
        "Some <mask> is defined as something that {w} regularly.",
    ],
    "item_vtran":[
        "The item is called as a <mask> because the main thing that it does is to {w} something.",
        "The item is called as an <mask> because the main thing that it does is to {w} something.",
        "The item is called as the <mask> because the main thing that it does is to {w} something .",
    ],
    "state_vintran": [
        "<mask> is the state of to {w}."
    ],
    "state_vtran": [
        "<mask> is the state of to {w} something."
    ],
    
    # Instrument
    
    "inst_vintran": [
        # "The item is called as a <mask> because the function of it is to {w}.",
        # "The item is called as an <mask> because the main thing that it does is to {w}.",
        # "The item is called as the <mask> because the main thing that it does is to {w}."

        "Technically, <mask> is defined as when someone or something use it to {w} ."
    ],
    "inst_vtran": [
        # "The item is called as a <mask> because it is the instrument to {w}.",
        "The item is called as a <mask> because the main thing that it does is to {w} something.",
        "The item is called as an <mask> because the main thing that it does is to {w} something.",
        "The item is called as the <mask> because the main thing that it does is to {w} something."
        # "A <mask> is needed to {w}",
        # "The <mask> is needed to {w}."
    ],

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
            token = tokenizer.decode(idx).strip().lower()
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