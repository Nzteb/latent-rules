Data downloaded from https://github.com/liu-yushan/PoLo/tree/main/datasets/Hetionet
and renamed as explained below:
NEW NAME = OLD NAME

valid.txt = dev.txt 
train.txt = graph.txt (which contains train.txt)
test.txt = test.txt

train-target.txt = train.txt (the subset of graph.txt triples that use the target relation CtD)

Not used:
rules.txt (rules ins special format used as input to the method Polo)
graph_inverses.txt seems to be the inverse equivalent of graph.txt
