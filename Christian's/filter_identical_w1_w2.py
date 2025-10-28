import sys

for l in sys.stdin:
    w1, w2, score = l.strip().split()
    if w1 != w2:
        print(l.strip())
