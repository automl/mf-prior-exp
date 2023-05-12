# Clean the priors directory before running
```bash
rm /work/dlclarge1/mallik-mf-prior/mf-prior-exp/src/mf-prior-bench/priors/*
```
# LCBench
EDIT: I just run them sequentially now
Seems like running all the lcbenchs in parallel is causing some issues.
My guess is trying to run YAHPO-gym in parallel with its file based config
is causing some race conditions are some weirness to happen.

Solution is run and just try those ones again

```bash
find ./logs -name "*.log" -exec cat {} \;
# Take note of ones that failed using "Array Task: <number>
sbatch --array=<number>,<number>,... generate_priors.sh
```
