import argparse, torch
from transformers import AutoTokenizer, RobertaForMaskedLM


MASK_ID = 50264

TEMPLATES = {
    "adj": "One of them is very <mask>.",
    "noun": "The child saw the <mask>.",
    "vintrans": "One of them will <mask>.",
    "vtrans": "One of them will <mask> another one."
}
    #"noun": "This is the <mask>.",
    #"noun": "Here is a <mask>.",
    #"noun": "I saw the <mask>.",

argparser = argparse.ArgumentParser(
    """
    Extract likely predicates from RoBERTa.
    """
)

argparser.add_argument(
    "pos", choices=["adj", "noun", "vintrans", "vtrans"],
    help="part of speech"
)

argparser.add_argument(
    "-n", type=int, default=50,
    help="number of predicates"
)

argparser.add_argument(
    "-a", "--alphabetize", default=False, action="store_true",
    help="alphabetize results"
)

# TODO option for model?
#    argparser.add_argument(
#        "-m",
#        "--model",
#        default="125m"
#    )

args = argparser.parse_args()
pos = args.pos

# location of models: /home/clark.3664/.cache/huggingface/hub/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
template = TEMPLATES[pos]

model_input = tokenizer(template, return_tensors="pt")
input_ids = model_input.input_ids[0]
mask_index = input_ids.tolist().index(MASK_ID)
logits = model(**model_input).logits
mask_logits = logits[0, mask_index]
sorted_vals, sorted_indices = torch.sort(mask_logits, descending=True)

words = list()
for ix in sorted_indices:
    word = tokenizer.decode(ix).strip()
    if not word.isalpha(): continue
    words.append(word)
    if len(words) == args.n: break

if args.alphabetize:
    words = sorted(words)

for word in words:
    print(word)

