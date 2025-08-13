import argparse, torch
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, RobertaForMaskedLM


MASK_ID = 50264

TEMPLATES = {
    "adj": [
        "He looks <mask>.", 
        "It is really <mask>.",
        "One of them is very <mask>.",
        "This looks <mask>.",
        "This is a <mask> dog."
    ],
    "noun": [
        "The child saw the <mask>.",
        "This is the <mask>.",
        "I saw a <mask> today.",
        "He looks like <mask>."
    ],
    "vintrans": [
        "One of them will <mask>.",
        "They often <mask>.",
        "People can <mask>."
    ],
    "vtrans": [
        "One of them will <mask> another one.",
        "I will <mask> it.",
        "They want to <mask> something."
    ],
    "particle": [
        "Sequence made <mask> of numbers are also called progressions.",
        "Sometimes companies try to find <mask> what people want.",
        "A sentence is a group of words that are put <mask> to express a complete thought.",
        "Clay is a fine-grained silicate mineral made when rocks break <mask>.",
        "The RNA strand than takes <mask> the cell machinery to make proteins."
    ],
    "prep": [
        "The book is <mask> the table.",
        "She walked <mask> the room.",
        "He lives <mask> New York.",
        "I will meet you <mask> Monday.",
        "The cat jumped <mask> the box."
    ], 
    "compadj":[
        "This movie is <mask> than the last one.",
        "He is much <mask> than his brother.", 
        "I think this solution is <mask>.", 
        "She runs <mask> than me.", 
        "The weather is getting <mask> every day."
    ],
    "adv":[
        "He runs <mask>.",
        "She spoke <mask> to the audience.",
        "They often arrive <mask>.",
        "I really like it because it is <mask>."
    ],
    "aux":[
        "He <mask> go to the store.",
        "They <mask> be finished by now.",
        "I <mask> help you.",
        "You <mask> have seen it."
    ],
    "determiner":[
        "<mask> cat is sleeping on the bed.",
        "I saw <mask> people in the park.",
        "We need <mask> solution.",
        "<mask> of the students passed the exam."
    ],
    "pronoun":[
        "Alex is a runner. <mask> is running fast.",
        "I don't know where Alex is today, but I saw <mask> yesterday.",
        "Alex has no work now. <mask> will finish the task.",
        "Alex is not here yet. We are waiting for <mask>."
    ],
    "conj":[
        "I like apples <mask> oranges.",
        "He left early <mask> he was tired.",
        "We will go out <mask> it rains.",
        "You can choose tea <mask> coffee."
    ],
    "interjection":[
        "<mask>! I forgot my keys.",
        "<mask>, I think you are right.",
        "<mask>, that is a great idea.",
        "<mask>, I don’t agree with you."
    ],
    "num":[
        "I have <mask> apples.",
        "She is the <mask> person in line.",
        "The room contains <mask> chairs.",
        "He finished in <mask> place."
    ],
    "negation":[
        "I will <mask> forget this.",
        "He does <mask> like broccoli.",
        "We have <mask> seen such a thing.",
        "It is <mask> too late to try."
    ],
    "superlative":[
        "She is the <mask> student in the class.",
        "This is the <mask> day of my life.",
        "He runs the <mask> among all the players.",
        "It is the <mask> solution we have found.",
        "That was the <mask> movie I have ever seen."
    ]
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

# location of models: /home/clark.3664/.cache/huggingface/hub/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
lemmatizer = WordNetLemmatizer()
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

        # Filtering rules
        if pos == "noun":
            # Only singular nouns
            lemmatized = lemmatizer.lemmatize(word, pos='n')
            if lemmatized != word:
                continue
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

