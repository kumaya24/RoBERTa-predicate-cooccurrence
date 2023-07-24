import sys, nltk, re
from nltk.stem import WordNetLemmatizer

COLLAPSE_UNARY = False

target_pos = sys.argv[1]
assert target_pos in ["adj", "noun", "vtrans", "vintrans"]
lemmatizer = WordNetLemmatizer()

words = set()

# stdin is senttrees
for t in sys.stdin:
    t = t.strip()
    if t == "!ARTICLE": continue
    t = nltk.tree.Tree.fromstring(t)
    if COLLAPSE_UNARY:
        t.collapse_unary(collapsePOS=True)
    for w, pos in t.pos():
        if COLLAPSE_UNARY:
            # when unary chains are collapsed, the syntactic categories
            # get combined like NP+PRP. This selects the top category in
            # the unary chain
            pos = pos.split("+")[0]
        # skip trace (e.g. *T*-1) and *NULL*
        #if not re.match("\*T\*-\d|\*NULL\*", w):
        if "*" in w: continue
        elif target_pos == "adj" and pos == "A-aN":
            words.add(lemmatizer.lemmatize(w.lower(), pos='a'))
        elif target_pos == "noun" and pos == "N-aD":
            words.add(lemmatizer.lemmatize(w.lower(), pos='n'))
        elif target_pos == "vtrans" and pos == "V-aN-bN" \
            or target_pos == "vintrans" and pos == "V-aN":
            words.add(lemmatizer.lemmatize(w.lower(), pos='v'))

for w in sorted(words):
    print(w)
