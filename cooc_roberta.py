import torch, sys, math, argparse, logging
from collections import defaultdict
from scipy.stats import entropy as KL
from torch.nn.functional import softmax
from transformers import AutoTokenizer, RobertaForMaskedLM

VOCAB_SIZE = 50265
MASK_ID = 50264

DEBUG = True
def eprint(*args, **kwargs):
    if DEBUG:
        print(*args, file=sys.stderr, **kwargs)

# TODO better template for noun-noun? desired properties:
# - works with countable (e.g. desk) and uncountable nouns (e.g. milk)
# - works with vowel-initial and consonant-initial nouns
SHORT_TEMPLATES = {
    "adj": {
        "adj": "He is <1> and <2>.",
        "noun": "This is a <1> <2> .",
        "vintrans": "The one who is <1> will <2>.",
        "vtrans": "The one who is <1> will <2> something.",
        "particle": "He is so <1> <2> he is dead."
    },
    "noun": {
        "noun": "A <1> is a kind of <2>.",
        "vintrans": "The <1> will <2> soon.",
        "vtrans": "Someone will <2> the <1>."
    },
    "vintrans": {
        "vintrans": "They often <1> and <2>.",
        "vtrans": "One of them will <1> and then <2> something."
    },
    "vtrans": {
        "vtrans": "They will <1> and <2> something.",
        "vintrans": "Someone who can <1> can also <2>."
    },
    "particle": {
        "particle": "He wondered <1> and <2> it would happen."
    },
    "prep": {
        "prep": "The book is <1> the box and <2> the chair."
    },
    "compadj": {
        "compadj": "He is <1> than her and <2> than me.",
        "noun": "A <1> person than most is also a <2> one."
    },
    "adv": {
        "adv": "He runs <1> and <2>.",
        "vtrans": "She can <1> quickly and <2> carefully."
    },
    "aux": {
        "aux": "He <1> go now and <2> later."
    },
    "determiner": {
        "determiner": "<1> and <2> cats are outside."
    },
    "pronoun": {
        "pronoun": "<1> and <2> are late to class."
    },
    "conj": {
        "conj": "I want apples <1> oranges, not <2> bananas."
    },
    "interjection": {
        "interjection": "<1>! That’s what I said. <2>! That’s what I meant."
    },
    "num": {
        "num": "I saw <1> and <2> ducks in the pond."
    },
    "negation": {
        "negation": "He will <1> and <2> do that again."
    },
    "superlative": {
        "superlative": "She is the <1> and <2> of all.",
        "adj": "This is the <1> and most <2> result."
    }
}


PROPOSAL_TEMPLATES = {
    "adj": {
        "adj": "Often a thing which is <1> is also a thing which is <2>.",
        "noun": "Often a thing which is <1> is also a <2>.",
        "vintrans": "Often a thing which is <1> is also a thing which can <2>.",
        "vtransObj": "Often a thing which is <1> is also a thing which another thing can <2>.",
        "vtransSubj": "Often a thing which is <1> is also a thing which can <1> another thing.",
        "particle": "They are <1> and not knowing <2> the death of Connor."
    },
    "noun": {
        "noun": "Often a <1> is also a <2>.",
        "vintrans": "Often a <1> is also a thing which can <2>.",
        "vtransObj": "Often a <1> is also a thing which another thing can <2>.",
        "vtransSubj": "Often a <1> is also a thing which can <2> another thing."
    },
    "vintrans": {
        "vintrans": "Often a thing which can <1> is also a thing which can <2>.",
        "vtransObj": "Often a thing which can <1> is also a thing which another thing can <2>.",
        "vtransSubj": "Often a thing which can <1> is also a thing which can <2> another thing."
    },
    "vtransObj": {
        "vtransObj": "Often a thing which another thing can <1> is also a thing which another thing can <2>.",
        "vtransSubj": "Often a thing which another thing can <1> is also a thing which can <2> another thing."
    },
    "vtransSubj": {
        "vtransSubj": "Often a thing which can <1> another thing is also a thing which can <2> another thing."
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


def single_batch_probability(
        sents, target_ids, target_start_ix, tokenizer, model
    ):
    try:
        model_input = tokenizer(sents, return_tensors="pt")
    except:
        eprint("Tokenization error. Sentences:")
        for s in sents:
            eprint(s)
            input_ids = tokenizer(s, return_tensors="pt").input_ids
            eprint("Input IDs shape:", input_ids.shape)
        raise
  
    logits = model(**model_input).logits
    probs = softmax(logits, dim=-1) #.detach().numpy()
    target_probs = torch.ones(len(sents))
    tok_count = target_ids.shape[1]
    for i in range(tok_count):
        ix = target_start_ix + i
        target_id_slice = target_ids[:, i].unsqueeze(dim=-1)
        probs_slice = probs[:, ix, :]
        cur_tok_probs = torch.gather(probs_slice, 1, target_id_slice).squeeze()
        target_probs *= cur_tok_probs

    return target_probs.detach().numpy()


def print_probabilities(
        template, words1, words1type, words1_tok_count,
        words2, words2type, words2_tok_count,
        target, tokenizer, model, batch_size=50
    ):
    if target == 1:
        tok_input = [" "+w for w in words1]
        target_ids = tokenizer(tok_input, return_tensors="pt").input_ids[:, 1:-1]
        target_ids = target_ids.repeat_interleave(len(words2), 0)
        mask = " ".join(["<mask>"]*words1_tok_count)
        masked_sents = list()
        for w1 in words1:
            for w2 in words2:
                sent = template
                # TODO these lines are for SHORT_TEMPLATES
                # hacky way of dealing with vowel-initial nouns that need
                # "an" for their determiner
                if words1type == "noun" and words2type == "noun" and w1[0] in 'aeiou':
                    sent = sent.replace("A <1>", "An <1>")
                if words1type == "noun" and words2type == "noun" and w2[0] in 'aeiou':
                    sent = sent.replace("a <2>", "an <2>")

                # TODO these lines are for PROPOSAL_TEMPLATES
                #if words1type == "noun" and w1[0] in 'aeiou':
                #    sent = sent.replace("a <1>", "an <1>")
                #if words2type == "noun" and w2[0] in 'aeiou':
                #    sent = sent.replace("a <2>", "an <2>")

                sent = sent.replace("<1>", mask).replace("<2>", w2)
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
            for w2 in words2:
                sent = template
                # TODO these lines are for SHORT_TEMPLATES
                if words1type == "noun" and words2type == "noun" and w1[0] in 'aeiou':
                    sent = sent.replace("A <1>", "An <1>")
                if words1type == "noun" and words2type == "noun" and w2[0] in 'aeiou':
                    sent = sent.replace("a <2>", "an <2>")

                # TODO these lines are for PROPOSAL_TEMPLATES
                #if words1type == "noun" and w1[0] in 'aeiou':
                #    sent = sent.replace("a <1>", "an <1>")
                #if words2type == "noun" and w2[0] in 'aeiou':
                #    sent = sent.replace("a <2>", "an <2>")

                sent = sent.replace("<1>", w1).replace("<2>", mask)
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

    # case 1: each batch is a single w1 paired with a subset of w2s
    if len(words2) > batch_size:
        logging.info("words2 is bigger than batch size")
        batches_per_w1 = 1 + (len(words2)-1) // batch_size
        for i, w1 in enumerate(words1):
            logging.info("word1: {}".format(w1))
            masked_sents_w1 = masked_sents[i*len(words2):(i+1)*len(words2)]
            target_ids_w1 = target_ids[i*len(words2):(i+1)*len(words2)]
            for b in range(batches_per_w1):
                logging.info("Batch {}/{}".format(b+1, batches_per_w1))
                start_ix = b * batch_size
                end_ix = (b+1) * batch_size
                batch_words2 = words2[start_ix:end_ix]
                batch_masked_sents = masked_sents_w1[start_ix:end_ix]
                batch_target_ids = target_ids_w1[start_ix:end_ix]
                batch_probs = single_batch_probability(
                    batch_masked_sents, batch_target_ids, target_start_ix,
                    tokenizer, model
                )
                for j, w2 in enumerate(batch_words2):
                    prob = batch_probs[j]
                    print("\t".join(
                        [w1, words1type, str(words1_tok_count),
                        w2, words2type, str(words2_tok_count),
                        template, str(target), str(prob)]
                    ))

    # case 2: each batch is one or more w1s, each paired with every possible w2
    else:
        logging.info("words2 fits into a batch")
        w1s_per_batch = batch_size // len(words2)
        actual_batch_size = w1s_per_batch * len(words2)
        assert w1s_per_batch >= 1
        num_batches = 1 + (len(words1)-1) // w1s_per_batch
        for b in range(num_batches):
            batch_words1 = words1[b*w1s_per_batch:(b+1)*w1s_per_batch]
            logging.info("Batch {}/{}".format(b+1, num_batches))
            logging.info("words1 batch: {}".format(batch_words1))

            start_ix = b * actual_batch_size
            end_ix = (b+1) * actual_batch_size
            batch_masked_sents = masked_sents[start_ix:end_ix]
            batch_target_ids = target_ids[start_ix:end_ix]
            batch_probs = single_batch_probability(
                batch_masked_sents, batch_target_ids, target_start_ix,
                tokenizer, model
            )
            for i, prob in enumerate(batch_probs):
                w1_ix = i // len(words2)
                w1 = batch_words1[w1_ix]
                w2_ix = i % len(words2)
                w2 = words2[w2_ix]
                print("\t".join(
                    [w1, words1type, str(words1_tok_count),
                    w2, words2type, str(words2_tok_count),
                    template, str(target), str(prob)]
                ))



def main():
    argparser = argparse.ArgumentParser(
        """
        Compute word similarity using Roberta, a masked LM.
        """
    )

    argparser.add_argument("words1")
    argparser.add_argument(
        "words1type",
        choices=list(SHORT_TEMPLATES.keys())
    )
    argparser.add_argument("words2")
    argparser.add_argument(
        "words2type",
        choices=list(SHORT_TEMPLATES.keys())
    )
    argparser.add_argument("-b", "--batch-size", type=int, default=50)
    argparser.add_argument("-l", "--logfile", default="log.txt")
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
    batch_size = args.batch_size

    logging.basicConfig(filename=args.logfile, level=logging.DEBUG)

    if words2type < words1type:
        words_temp = words1
        wordstype_temp = words1type
        words1 = words2
        words1type = words2type
        words2 = words_temp
        words2type = wordstype_temp


    templates = SHORT_TEMPLATES
    #templates = PROPOSAL_TEMPLATES
    template = templates[words1type][words2type]

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
            print_probabilities(
                template, words1_leni, words1type, i,
                words2_lenj, words2type, j, 
                target, tokenizer, model, batch_size
            )

    target = 2
    for i in words1_tok_counts:
        words1_leni = words1_by_tok_count[i]
        words1_leni.insert(0, " ".join(["<mask>"]*i))
        for j in words2_tok_counts:
            #print("####\ttargetWord:{}\tword1TokLen:{}\tword2TokLen:{}".format(target, i, j))
            words2_lenj = words2_by_tok_count[j][:]
            print_probabilities(
                template, words1_leni, words1type, i,
                words2_lenj, words2type, j, 
                target, tokenizer, model, batch_size
            )
                    

if __name__ == "__main__":
    main()

