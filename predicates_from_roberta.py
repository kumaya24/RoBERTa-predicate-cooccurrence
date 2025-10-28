import argparse, torch
# from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, RobertaForMaskedLM


MASK_ID = 50264

TEMPLATES = {
    "adj": [
        "They are <mask> ones."
    ],
    "noun": [
        # "The <mask> is what there is."
        #"The <mask> is something to discuss."
        # "The <mask> is what the child saw."
        "The <mask> was all the child saw."
    ],
    "vintrans": [
        # "They will <mask> soon.",
        # "They also will <mask> for it."
        "One of them will <mask>."
    ],
    "vtrans": [
        # "They will <mask> it.",
        # "It is something that they will <mask>."
        "One of them will <mask> it."
    ],
    "particleVerbUp": [
        "They will <mask> up."
    ],
    "particleVerbDown": [
        "They will <mask> down."
    ],
    "particleVerbIn": [
        "They will <mask> in."
    ],
    "particleVerbOut": [
        "They will <mask> out."
    ],
    "particleVerbOver": [
        "They will <mask> over."
    ],
    "particleVerbAlong": [
        "They will <mask> along."
    ],
    "particleVerbAway": [
        "They will <mask> away."
    ],
    "particleVerbTo": [
        "They will <mask> to."
    ],
    "particleVerbInto": [
        "They will <mask> into."
    ],
    "particleVerbOff": [
        "They will <mask> off."
    ],
    "particleVerbOn": [
        "They will <mask> on."
    ],
    "particleVerbAround": [
        # "They will <mask> around.",   # B-aN-bP...
        "This is something that they will <mask> around."  # B-aN-bP...-bN
    ],
    "particleVerbUnder": [
        "They will <mask> under."
    ],
    "prep": [
        "One thing is located <mask> another thing.",
        "One thing happens <mask> another thing."
    ], 
    "compadj":[
        "This one is <mask> than the other one.",
    ],
    "adv":[
        "He does it more <mask>."
        # "They often arrive <mask>.",
    ],
    "aux":[
        "They <mask> do it."
    ],
    "determiner":[
        "There is <mask> thing"
    ],
    "pronoun":[
        "Alex is a runner. <mask> is running fast.",
        "He said to me that <mask> are a good person.",
        "He said to me that <mask> are good people."
    ],
    "conj":[
        "I like this <mask> she likes that.",
        "I like this <mask> it is good.",
        "I can only do this <mask> that."
    ],
    "interjection":[
        "<mask>! I did it."
    ],
    "negation":[
        "This is <mask> anything.",
        "I would <mask> get anything from it.",
    ],
    "superlative":[
        "This is the <mask> thing of all.",
        "He is the <mask> among all of them.",
        "It is the <mask> thing we have done.",
        "That was the <mask> one I have ever seen."
    ],
    "complementClauseVerb": [
        "I will <mask> that it is something."
    ],
    "complementClauseNoun": [
        "The <mask> that I do is good."
    ],
    "nullComplementClauseVerb": [
        "I will <mask> it is true."
    ],
    "nullComplementClauseNoun": [
        "I have heard about <mask> that they did it.",
        "The <mask> that they did it was surprising."
    ],
     "baseComplementClauseVerb": [
        "They will <mask> that the truth is out there."
    ],
    "baseComplementClauseNoun": [
        "They know the <mask> that the truth is out there."
    ],
    "infinitivalComplementClauseVerb": [
        "They will <mask> to improve it."
    ],
    "infinitivalComplementClauseNoun": [
        "They made the <mask> to do it."
    ],
    "raisingVerb": [
        "They did <mask> to like it."
    ],
    "gerundVerb": [
        "They like go <mask>.",
        "Their favorite activity is <mask>."
    ]
}

argparser = argparse.ArgumentParser(
    """
    Extract likely predicates from RoBERTa.
    """
)

argparser.add_argument(
    "pos", choices=list(TEMPLATES.keys()),
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

argparser.add_argument(
    "-s", "--scores", default=False, action="store_true",
    help="output logit scores"
)

# TODO option for model?
#    argparser.add_argument(
#        "-m",
#        "--model",
#        default="125m"
#    )

args = argparser.parse_args()
pos = args.pos

# Try to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
# lemmatizer = WordNetLemmatizer()
model.to(device)
model.eval()  # inference mode
all_words = []
all_scores = []

for template in TEMPLATES[pos]:
    model_input = tokenizer(template, return_tensors="pt")
    input_ids = model_input.input_ids[0]
    mask_index = input_ids.tolist().index(MASK_ID)
    
    logits = model(**model_input).logits
    mask_logits = logits[0, mask_index]
    sorted_vals, sorted_indices = torch.sort(mask_logits, descending=True)

    for v, ix in zip(sorted_vals, sorted_indices):
        word = tokenizer.decode(ix).strip()
        if not word.isalpha(): 
            continue
        
        '''
        # Filtering rules
        if pos == "noun":
            # Only singular nouns
            lemmatized = lemmatizer.lemmatize(word, pos='n')
            if lemmatized != word:
                continue
        '''

        if pos in ["particle", "prep", "determiner", "pronoun", "negation"]:
            # Skip capitalized or long content words
            if word[0].isupper() or len(word) > 10:
                continue

        all_words.append(word)
        all_scores.append(v.item())
        if len(all_words) >= args.n:
            break

# Deduplicate while keeping highest score
word_score_dict = {}
for word, score in zip(all_words, all_scores):
    if word not in word_score_dict or score > word_score_dict[word]:
        word_score_dict[word] = score

if args.alphabetize:
    wordscores = sorted(word_score_dict.items(), key=lambda x: x[0])
else:
    wordscores = sorted(word_score_dict.items(), key=lambda x: x[1], reverse=True)

wordscores = wordscores[:args.n]

# Print results
for word, score in wordscores:
    if args.scores:
        print(f"{word}\t{score}")
    else:
        print(word)

