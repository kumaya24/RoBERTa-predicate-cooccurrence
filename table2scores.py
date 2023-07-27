import math, sys

words1 = set()
words2 = set()
scores_target_w1 = dict()
scores_target_w2 = dict()

# header
sys.stdin.readline()

# sys.stdin is the output from cooc_roberta.py
for l in sys.stdin:
    if l.startswith("#"): continue
    #_, _, _, w1, w2, target, value = l.split("\t")
    w1, _, w1_tok_len, w2, _, w2_tok_len, _, target, value = l.split('\t')
    w1_tok_len = int(w1_tok_len)
    w2_tok_len = int(w2_tok_len)
    target = int(target)
    value = float(value)
    if "<mask>" not in w1:
        words1.add((w1, w1_tok_len))
    if "<mask>" not in w2:
        words2.add((w2, w2_tok_len))
    if target == 1:
        scores_target_w1[(w1, w2)] = value
    else:
        assert target == 2, "bad target: {}".format(target)
        scores_target_w2[(w1, w2)] = value

words1 = list(sorted(words1, key=lambda x: x[0]))
words2 = list(sorted(words2, key=lambda x: x[0]))
print("word1\tword2\tscore")
for w1, w1_tok_len in words1:
    for w2, w2_tok_len in words2:
        w1_given_w2 = scores_target_w1[(w1, w2)]
        w2_mask = " ".join(["<mask>"]*w2_tok_len)
        w1_given_mask = scores_target_w1[(w1, w2_mask)]
        quasi_pmi_w1 = math.log(w1_given_w2 / w1_given_mask)

        w2_given_w1 = scores_target_w2[(w1, w2)]
        w1_mask = " ".join(["<mask>"]*w1_tok_len)
        w2_given_mask = scores_target_w2[(w1_mask, w2)]
        quasi_pmi_w2 = math.log(w2_given_w1 / w2_given_mask)
    
        # could also take the mean and/or not include floor value of 0
        score = max([quasi_pmi_w1, quasi_pmi_w2, 0])
        print(w1, w2, score)
