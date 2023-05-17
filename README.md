# MF + Priors

NOTE: For anonymity reasons, we can not release the link to download the surrogates for
the PD1 benchmarks.

## Installation

### 1. Clone the repository

**Removed for anon**

Unzip the archive.

### 2. Conda, Poetry, Package, Pre-Commit

To setup tooling and install the package, follow this documentation (**removed**) using the environment name of your choice.

**NOTE: Requires Python 3.7**

```bash
poetry install
```

### 3. Just

To install our command runner just you can check the [just documentation](https://github.com/casey/just#installation), or run the below command

```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to $HOME/.just
```

Also, to make the `just` command available you should add

```bash
export PATH="$HOME/.just:$PATH"
```

to your `.zshrc` / `.bashrc` or alternatively simply run the export manually.

### 4. Data

```bash
python -m mfpbench download
```

## Usage

### Running a singular local experiment

```bash
# We need at least two seeds for the plotting to work
just run random_search lcbench-189906_prior-at25 test 0
just run random_search lcbench-189906_prior-at25 test 1
just run priorband lcbench-189906_prior-at25 test 0
just run priorband lcbench-189906_prior-at25 test 1

# For more options see which can be run directly
python -m mf_prior_experiments.run -h
python -m mf_prior_experiments.run algorithm=random_search benchmark=lcbench189906_prior-at25 experiment_group=test
```

### Running a grid of experiments on slurm

To run 10 seeds for two algorithms and benchmarks, e.g.,

```bash
just submit random_search,priorband lcbench189906_prior-at25 "range(0, 5)" test
# note the whitespace in `"range(0, 5)"`.

# Use `just submit` to see the arguments
```

### Analysing experiments

All plots included are provided by `src/mf_prior_experiments/plot.py`. The plots
rely on the results being collected and cached first, before being plotted, you can
do so with.

```bash
python -m src.mf_prior_experiments.plot \
    --collect \
    --collect-ignore-missing \
    --experiment_group "test" \
    --n_workers 1  # If collecting results for parallel runs, specify the number of workers
    # --parallel  # To speed up collection of many results
```

To do the relative ranking plots:
```bash
# JSON in bash ... yup
incumbent_traces="
{
  \"Good prior\": [\"lcbench-189906_prior-at25\"]
}
"

python -m src.mf_prior_experiments.plot \
	--algorithms random_search priorband \
	--experiment_group "test" \
	--prefix "test_plot" \
	--dpi 200 \
	--ext "png" \
	--plot_default \
    --dynamic_y_lim \
	--n_workers 1 \
	--incumbent_traces "${incumbent_traces}" \
	--x_range_it 1 12 \
```
