import sys, argparse, math
from numpy.random import dirichlet, choice

CONCENTRATION = 1.5

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subjects", default=10, type=int, help="number of subjects")
parser.add_argument("-v", "--verbs",  default=10, type=int, help="number of verbs")
parser.add_argument("-o", "--objects", default=10, type=int, help="number of objects")
parser.add_argument("-n", "--sentences", default=100, type=int, help="number of sentences")
parser.add_argument("-c", "--corpus", default="corpus.txt", help="corpus file name")
parser.add_argument("-a", "--associations", default="associations.tsv", help="associations file name")

args = parser.parse_args()
num_subj = args.subjects
num_verb = args.verbs
num_obj = args.objects
num_sents = args.sentences
corpus = args.corpus
associations = args.associations

verb_prob = dirichlet([CONCENTRATION]*num_verb)
per_verb_subj_prob = list()
per_verb_obj_prob = list()

for i in range(num_verb):
    subj_prob = dirichlet([CONCENTRATION]*num_subj)
    per_verb_subj_prob.append(subj_prob)
    obj_prob = dirichlet([CONCENTRATION]*num_obj)
    per_verb_obj_prob.append(obj_prob)

with open(corpus, "w") as f:
    for i in range(num_sents):
        verb = choice(range(num_verb), p=verb_prob)
        subj_prob = per_verb_subj_prob[verb]
        subj = choice(range(num_subj), p=subj_prob)
        obj_prob = per_verb_obj_prob[verb]
        obj = choice(range(num_obj), p=obj_prob)
        f.write("s{} v{} o{}\n".format(subj, verb, obj))

# denominators P(S) and P(O) for PMI
subj_prob = list()
for s in range(num_subj):
    s_prob = 0
    for v in range(num_verb):
        pv = verb_prob[v]
        ps_given_v = per_verb_subj_prob[v][s]
        s_prob += pv * ps_given_v
    subj_prob.append(s_prob)

obj_prob = list()
for o in range(num_obj):
    o_prob = 0
    for v in range(num_verb):
        pv = verb_prob[v]
        po_given_v = per_verb_obj_prob[v][o]
        o_prob += pv * po_given_v
    obj_prob.append(o_prob)

with open(associations, "w") as f:
    f.write("pred1\tpred1Role\tpred2\tpred2Role\tscore\n")
    for v in range(num_verb):
        for s in range(num_subj):
            ps_given_v = per_verb_subj_prob[v][s]
            ps = subj_prob[s]
            assoc = math.log2(ps_given_v / ps)
            f.write("A{}\t{}\tV{}\t{}\t{}\n".format(s, 0, v, 1, assoc))
    for v in range(num_verb):
        for o in range(num_obj):
            po_given_v = per_verb_obj_prob[v][o]
            po = obj_prob[o]
            assoc = math.log2(po_given_v / po)
            f.write("P{}\t{}\tV{}\t{}\t{}\n".format(o, 0, v, 2, assoc))

