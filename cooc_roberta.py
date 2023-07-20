import torch, sys, math, argparse
from torch.nn.functional import softmax
from transformers import AutoTokenizer, RobertaForMaskedLM
from scipy.stats import entropy as KL

# https://github.com/huggingface/transformers/issues/18104#issuecomment-1465329549

VOCAB_SIZE = 50265
MASK_ID = 50264

DEBUG = False
def eprint(*args, **kwargs):
    if DEBUG:
        print(*args, file=sys.stderr, **kwargs)


def find_index(short_seq, long_seq):
    """
    Returns the index of the first appearance of short_seq within long_seq,
    or -1 if short_seq never appears.
    """
    index = -1
    for i in range(len(long_seq) - len(short_seq)):
        if long_seq[i:i+len(short_seq)] == short_seq:
            index = i
            break
    return index


def populate_template(template, word1, word2, masks1, masks2):
    mask_mask = template.replace("<1>", masks1).replace("<2>", masks2)
    w1_mask = template.replace("<1>", word1).replace("<2>", masks2)
    w1_w2 = template.replace("<1>", word1).replace("<2>", word2)
    eprint("word1:", word1)
    eprint("word2:", word2)
    eprint(mask_mask)
    eprint(w1_mask)
    eprint(w1_w2)
    return mask_mask, w1_mask, w1_w2


# TODO change to calculate PMI
def _calculate_ratio(mask_mask_str, w1_mask_str, w2ids, w2ix, tokenizer, model):
    mask_mask_inputs = tokenizer(mask_mask_str, return_tensors="pt")
    mask_mask_ids = mask_mask_inputs.input_ids[0].tolist()
    mask_mask_logits = model(**mask_mask_inputs).logits
    mask_mask_probs = softmax(mask_mask_logits, dim=-1).detach().numpy()
    mask_mask_prob = 0
    for i, tok in enumerate(w2ids):
        ix = w2ix + i
        mask_mask_prob = mask_mask_probs[0, ix, tok]

    w1_mask_inputs = tokenizer(w1_mask_str, return_tensors="pt")
    w1_mask_ids = w1_mask_inputs.input_ids[0].tolist()
    w1_mask_logits = model(**w1_mask_inputs).logits
    w1_mask_probs = softmax(w1_mask_logits, dim=-1).detach().numpy()
    w1_mask_prob = 0
    for i, tok in enumerate(w2ids):
        ix = w2ix + i
        w1_mask_prob = w1_mask_probs[0, ix, tok]

    return w1_mask_prob / mask_mask_prob


def get_association(word1, word1type, word2, word2type, tokenizer, model):
    # reorganize so word1type is never alphabetically after word2type
    # (reduces number of cases below)
    if word2type < word1type:
        temp = word1
        temptype = word1type
        word1 = word2
        word1type = word2type
        word2 = temp
        word2type = temptype

    w1ids = tokenizer(" "+word1, return_tensors="pt").input_ids[0][1:-1].tolist()
    w2ids = tokenizer(" "+word2, return_tensors="pt").input_ids[0][1:-1].tolist()
    masks1 = " ".join(["<mask>"]*len(w1ids))
    masks2 = " ".join(["<mask>"]*len(w2ids))

    # TODO change to match-case statements?
    if word1type == "adj":
        if word2type == "adj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "They are very <1> and <2>.",
                word1, word2, masks1, masks2
            )
        elif word2type == "noun":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "This is a very <1> <2>.",
                word1, word2, masks1, masks2
            )
        # TODO someone instead of something?
        elif word2type == "vintrans":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "Something very <1> will <2>.",
                word1, word2, masks1, masks2
            )
        elif word2type == "vtransObj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "They will <2> something very <1>.",
                word1, word2, masks1, masks2
            )
        elif word2type == "vtransSubj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "Something very <1> will <2> them.",
                word1, word2, masks1, masks2
            )
        else: raise

    elif word1type == "noun":
        # TODO deal with "an" for vowel-initial nouns
        if word2type == "noun":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "A <1> is a <2>.",
                word1, word2, masks1, masks2
            )
        elif word2type == "vintrans":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "The <1> will <2>.",
                word1, word2, masks1, masks2
            )
        elif word2type == "vtransObj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "They will <2> the <1>.",
                word1, word2, masks1, masks2
            )
        elif word2type == "vtransSubj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "The <1> will <2> them.",
                word1, word2, masks1, masks2
            )
        else: raise

    elif word1type == "vintrans":
        if word2type == "vintrans":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "They will <1> and <2>.",
                word1, word2, masks1, masks2
            )
        # TODO whoever instead of whatever?
        elif word2type == "vtransObj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "They will <2> whatever will <1>.",
                word1, word2, masks1, masks2
            )
        elif word2type == "vtransSubj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "They will <2> them and <1>.",
                word1, word2, masks1, masks2
            )
        else: raise

    elif word1type == "vtransObj":
        if word2type == "vtransObj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "Someone will <1> them and someone will <2> them.",
                word1, word2, masks1, masks2
            )
        elif word2type == "vtransSubj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "They will <2> it and the others will <1> them.",
                word1, word2, masks1, masks2
            )
        else: raise

    elif word1type == "vtransSubj":
        if word2type == "vtransSubj":
            mask_mask, w1_mask, w1_w2 = populate_template(
                "They will <1> them and <2> the others.",
                word1, word2, masks1, masks2
            )
        else: raise

    else: raise

    w1_w2_input = tokenizer(w1_w2, return_tensors="pt")
    w1_w2_ids = w1_w2_input.input_ids[0].tolist()
    w2ix = find_index(w2ids, w1_w2_ids)

    # TODO calculate PMI instead
    return _calculate_ratio(mask_mask, w1_mask, w2ids, w2ix, tokenizer, model)



def main():
    argparser = argparse.ArgumentParser(
        """
        Compute word similarity using Roberta, a masked LM.
        """
    )

    argparser.add_argument("words1")
    argparser.add_argument(
        "words1type",
        choices=["noun", "adj", "vtransSubj", "vtransObj", "vintrans"]
    )
    argparser.add_argument("words2")
    argparser.add_argument(
        "words2type",
        choices=["noun", "adj", "vtransSubj", "vtransObj", "vintrans"]
    )
    # TODO option for model?
#    argparser.add_argument(
#        "-m",
#        "--model",
#        default="125m"
#    )
    args = argparser.parse_args()


    # location of models: /home/clark.3664/.cache/huggingface/hub/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")

    words1 = open(args.words1).readlines()
    words1 = [w.strip() for w in words1]
    words1type = args.words1type
    words2 = open(args.words2).readlines()
    words2 = [w.strip() for w in words2]
    words2type = args.words2type

    print("word1\word2\t" + "\t".join(words2))

    for w1 in words1:
        row = w1
        for w2 in words2:
            assoc = get_association(
                w1, words1type, w2, words2type, tokenizer, model
            )
            row += "\t" + str(round(assoc, 2))
        print(row)
        
            

if __name__ == "__main__":
    main()

