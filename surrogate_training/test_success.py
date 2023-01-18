from pathlib import Path
import mfpbench

DATADIR = "/work/dlclarge1/mallik-mf-prior/mf-prior-exp/data/pd1-data"

b = mfpbench.get("imagenet_resnet_512", datadir=DATADIR)
b.load()

s = b.sample()
print(s)

r = b.query(s)
print(r)

b = mfpbench.get("cifar100_wideresnet_2048", datadir=DATADIR)
b.load()

s = b.sample()
print(s)

r = b.query(s)
print(r)
