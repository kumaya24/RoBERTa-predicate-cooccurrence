import sys

pos1 = sys.argv[2]
role1 = sys.argv[3]
pos2 = sys.argv[4]
role2 = sys.argv[5]

print("pred1\tpred1Role\tpred2\tpred2Role\tscore")
f = open(sys.argv[1])
f.readline()
for l in f:
    w1, w2, score = l.strip().split()
    print("{}_{}\t{}\t{}_{}\t{}\t{}".format(
        w1, pos1, role1, w2, pos2, role2, score
    ))
