from pathlib import Path
import mfpbench

DATADIR = "/work/dlclarge1/mallik-mf-prior/mf-prior-exp/data/pd1-data"

b = mfpbench.get("lm1b_transformer_2048", datadir=DATADIR, prior="good")
b.load()

s = b.sample()
print(s)

r = b.query(s)
print(r)
