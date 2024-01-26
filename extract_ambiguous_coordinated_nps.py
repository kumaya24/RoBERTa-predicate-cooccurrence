import sys
from nltk.tree import Tree

for l in sys.stdin:
    t = Tree.fromstring(l.strip())
    for subt in t.subtrees():
        if subt.label() != "NP":
            continue
        pos = [x[1] for x in subt.pos()]
        lvs = subt.leaves()
        if pos == ["JJ", "NN", "CC", "NN"] and lvs[2] == "and":
        #if pos == ["DT", "NN"]:
            print(subt)

