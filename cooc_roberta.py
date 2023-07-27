import torch, sys, math, argparse
from collections import defaultdict
from scipy.stats import entropy as KL
from torch.nn.functional import softmax
from transformers import AutoTokenizer, RobertaForMaskedLM

# https://github.com/huggingface/transformers/issues/18104#issuecomment-1465329549

VOCAB_SIZE = 50265
MASK_ID = 50264

DEBUG = True
def eprint(*args, **kwargs):
    if DEBUG:
        print(*args, file=sys.stderr, **kwargs)

# TODO someone instead of something?
# TODO deal with "an" for vowel-initial nouns
# TODO whoever instead of whatever?
TEMPLATES = {
    "adj": {
        "adj": "They are very <1> and <2>.",
        "noun": "This is a very <1> <2>.",
        "vintrans": "Something very <1> will <2>.",
        "vtransObj": "They will <2> something very <1>.",
        "vtransSubj": "Something very <1> will <2> them."
    },
    "noun": {
        "noun": "A <1> is a <2>.",
        "vintrans": "The <1> will <2>.",
        "vtransObj": "They will <2> the <1>.",
        "vtransSubj": "The <1> will <2> them."
    },
    "vintrans": {
        "vintrans": "They will <1> and <2>.",
        "vtransObj": "They will <2> whatever will <1>.",
        "vtransSubj": "They will <2> them and <1>."
    },
    "vtransObj": {
        "vtransObj": "Someone will <1> them and someone will <2> them.",
        "vtransSubj": "They will <2> it and the others will <1> them."
    },
    "vtransSubj": {
        "vtransSubj": "They will <1> them and <2> the others."
    }
}


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


def prob_from_template(template, w1, w2, target, tokenizer, model):
    assert "<1>" in template and "<2>" in template

    if target == 1:
        target_ids = tokenizer(" "+w1, return_tensors="pt").input_ids[0][1:-1].tolist()
        masks = " ".join(["<mask>"]*len(target_ids))
        masked_sent = template.replace("<1>", masks).replace("<2>", w2)
    elif target == 2:
        target_ids = tokenizer(" "+w2, return_tensors="pt").input_ids[0][1:-1].tolist()
        masks = " ".join(["<mask>"]*len(target_ids))
        masked_sent = template.replace("<1>", w1).replace("<2>", masks)
    else:
        raise ValueError("Target must be 1 (word 1) or 2 (word 2)")

    # unmasked sentence is used to find the location of the target word
    # in the sequence of token ids
    unmasked_sent = template.replace("<1>", w1).replace("<2>", w2)
    unmasked_input = tokenizer(unmasked_sent, return_tensors="pt")
    unmasked_ids = unmasked_input.input_ids[0].tolist()
    target_start_ix = find_index(target_ids, unmasked_ids)

    masked_input = tokenizer(masked_sent, return_tensors="pt")
    logits = model(**masked_input).logits
    probs = softmax(logits, dim=-1).detach().numpy()
    target_word_prob = 1
    for i, tok in enumerate(target_ids):
        ix = target_start_ix + i
        target_word_prob *= probs[0, ix, tok]
    return target_word_prob


def batch_probabilities(
        template, words1, words1_tok_count, words2, words2_tok_count,
        target, tokenizer, model
    ):
    if target == 1:
        tok_input = [" "+w for w in words1]
        target_ids = tokenizer(tok_input, return_tensors="pt").input_ids[:, 1:-1]
        target_ids = target_ids.repeat_interleave(len(words2), 0)
        mask = " ".join(["<mask>"]*words1_tok_count)
        masked_sents = list()
        for _ in words1:
            for w2 in words2:
                sent = template.replace("<1>", mask).replace("<2>", w2)
                masked_sents.append(sent)
        # reverse-masked sentence is used to find the location of the target
        # word (word 1) in the sequence of token ids.  word2 is masked so that
        # won't be found if word1 and word2 are the same
        w2_mask = " ".join(["<mask>"]*words2_tok_count)
        reverse_masked_sent = template.replace("<1>", words1[0]).replace("<2>", w2_mask)
        reverse_masked_input = tokenizer(reverse_masked_sent, return_tensors="pt")
        reverse_masked_ids = reverse_masked_input.input_ids[0].tolist()
        target_start_ix = find_index(target_ids[0].tolist(), reverse_masked_ids)

    elif target == 2:
        tok_input = [" "+w for w in words2]
        target_ids = tokenizer(tok_input, return_tensors="pt").input_ids[:, 1:-1]
        # this is different from target=1 so that the results will still be
        # ordered by word1 and then word2
        target_ids = target_ids.tile(len(words1), 1)
        mask = " ".join(["<mask>"]*words2_tok_count)
        masked_sents = list()
        for w1 in words1:
            for _ in words2:
                sent = template.replace("<1>", w1).replace("<2>", mask)
                masked_sents.append(sent)
        # reverse-masked sentence is used to find the location of the target
        # word (word 2) in the sequence of token ids.  word1 is masked so that
        # won't be found if word1 and word2 are the same
        w1_mask = " ".join(["<mask>"]*words1_tok_count)
        reverse_masked_sent = template.replace("<1>", w1_mask).replace("<2>", words2[0])
        reverse_masked_input = tokenizer(reverse_masked_sent, return_tensors="pt")
        reverse_masked_ids = reverse_masked_input.input_ids[0].tolist()
        target_start_ix = find_index(target_ids[0].tolist(), reverse_masked_ids)

    else:
        raise ValueError("Target must be 1 (word 1) or 2 (word 2)")

    masked_input = tokenizer(masked_sents, return_tensors="pt")
    logits = model(**masked_input).logits
    probs = softmax(logits, dim=-1) #.detach().numpy()
    target_word_probs = torch.ones(len(masked_sents))
    for i in range(target_ids.shape[1]):
        ix = target_start_ix + i
        target_id_slice = target_ids[:, i].unsqueeze(dim=-1)
        probs_slice = probs[:, ix, :]
        cur_tok_probs = torch.gather(probs_slice, 1, target_id_slice).squeeze()
        target_word_probs *= cur_tok_probs
    # shape: len(words1) x len(words2)
    target_word_probs = target_word_probs.reshape([len(words1), len(words2)])
    return target_word_probs.detach().numpy()
    


#def quasi_pmi(
#        mask_mask, w1_mask, mask_w2, w1_w2,
#        w1ids, w2ids, tokenizer, model
#    ):
#    w1_w2_input = tokenizer(w1_w2, return_tensors="pt")
#    w1_w2_ids = w1_w2_input.input_ids[0].tolist()
#    w1ix = find_index(w1ids, w1_w2_ids)
#    w2ix = find_index(w2ids, w1_w2_ids)
#
#    mask_mask_inputs = tokenizer(mask_mask, return_tensors="pt")
#    mask_mask_logits = model(**mask_mask_inputs).logits
#    mask_mask_probs = softmax(mask_mask_logits, dim=-1).detach().numpy()
#    pr_w1_given_mask = 1
#    for i, tok in enumerate(w1ids):
#        ix = w1ix + i
#        pr_w1_given_mask *= mask_mask_probs[0, ix, tok]
#    pr_w2_given_mask = 1
#    for i, tok in enumerate(w2ids):
#        ix = w2ix + i
#        pr_w2_given_mask *= mask_mask_probs[0, ix, tok]
#
#    mask_w2_inputs = tokenizer(mask_w2, return_tensors="pt")
#    mask_w2_logits = model(**mask_w2_inputs).logits
#    mask_w2_probs = softmax(mask_w2_logits, dim=-1).detach().numpy()
#    pr_w1_given_w2 = 1
#    for i, tok in enumerate(w1ids):
#        ix = w1ix + i
#        pr_w1_given_w2 *= mask_w2_probs[0, ix, tok]
#
#    w1_mask_inputs = tokenizer(w1_mask, return_tensors="pt")
#    w1_mask_logits = model(**w1_mask_inputs).logits
#    w1_mask_probs = softmax(w1_mask_logits, dim=-1).detach().numpy()
#    pr_w2_given_w1 = 1
#    for i, tok in enumerate(w2ids):
#        ix = w2ix + i
#        pr_w2_given_w1 *= w1_mask_probs[0, ix, tok]
#
#    pmi_w1 = math.log(pr_w1_given_w2 / pr_w1_given_mask)
#    pmi_w2 = math.log(pr_w2_given_w1 / pr_w2_given_mask)
#
#    # could also take the mean and/or not include floor value of 0
#    return max([pmi_w1, pmi_w2, 0])



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

    if words1type < words2type:
        template = TEMPLATES[words1type][words2type]
    else:
        template = TEMPLATES[words2type][words1type]

    eprint("Sorting words by token count...")
    words1_by_tok_count = defaultdict(list)
    for w in words1:
        toks = tokenizer(" "+w, return_tensors="pt").input_ids[0][1:-1]
        words1_by_tok_count[len(toks)].append(w)
    words1_tok_counts = list(sorted(words1_by_tok_count.keys()))

    words2_by_tok_count = defaultdict(list)
    for w in words2:
        toks = tokenizer(" "+w, return_tensors="pt").input_ids[0][1:-1]
        words2_by_tok_count[len(toks)].append(w)
    words2_tok_counts = list(sorted(words2_by_tok_count.keys()))


    eprint("Computing cooccurrence probabilities...")
    #print("w1type\tw2type\ttemplate\tw1\tw2\ttarget\tvalue")
    print("w1\tw1type\tw1tokLen\tw2\tw2type\tw2tokLen\ttemplate\ttarget\tvalue")

    target = 1
    for i in words1_tok_counts:
        words1_leni = words1_by_tok_count[i]
        for j in words2_tok_counts:
            #print("####\ttargetWord:{}\tword1TokLen:{}\tword2TokLen:{}".format(target, i, j))
            words2_lenj = words2_by_tok_count[j][:]
            words2_lenj.insert(0, " ".join(["<mask>"]*j))
            probs = batch_probabilities(
                template, words1_leni, i, words2_lenj, j, 
                target, tokenizer, model
            )
            for k, w1 in enumerate(words1_leni):
                for l, w2 in enumerate(words2_lenj):
                    prob = probs[k, l]
                    print("\t".join(
                        [w1, words1type, str(i), w2, words2type, str(j),
                        template, str(target), str(prob)]
                    ))
                    #print("\t".join(
                    #    [words1type, words2type, template, 
                    #     w1, w2, str(target), str(prob)]
                    #))

    target = 2
    for i in words1_tok_counts:
        words1_leni = words1_by_tok_count[i]
        words1_leni.insert(0, " ".join(["<mask>"]*i))
        for j in words2_tok_counts:
            #print("####\ttargetWord:{}\tword1TokLen:{}\tword2TokLen:{}".format(target, i, j))
            words2_lenj = words2_by_tok_count[j][:]
            probs = batch_probabilities(
                template, words1_leni, i, words2_lenj, j, 
                target, tokenizer, model
            )
            for k, w1 in enumerate(words1_leni):
                for l, w2 in enumerate(words2_lenj):
                    prob = probs[k, l]
                    print("\t".join(
                        [w1, words1type, str(i), w2, words2type, str(j),
                        template, str(target), str(prob)]
                    ))
                    #print("\t".join(
                    #    [words1type, words2type, template, 
                    #     w1, w2, str(target), str(prob)]
                    #))
                    

if __name__ == "__main__":
    main()

