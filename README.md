# PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning

This repo is purely for reproducing the experiments of the paper. To use `PriorBand`, we have
an up to date implementation in [`NePs`][https://github.com/automl/neps] that should be the preffered
way to use the optimizer.

## Installation

### 1. Clone the repository

```bash
git clone --recursive https://github.com/automl/mf-prior-exp.git

# To use the version used for the paper
git clone --branch "vPaper-arxiv" --recursive https://github.com/automl/mf-prior-exp.git

# If you forgot to use the --recursive flag or checkout the branches, you can do the following
# to manuall perform the steps
cd mf-prior-exp

# If repeating experiments from the paper
# git checkout vPaper-priorband

git submodule update --init
```

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

You should get a layout that looks like so:

```
data
├── jahs-bench-data
│   ├── assembled_surrogates
│   │   ├── cifar10
│   │   ├── colorectal_histology
│   │   └── fashion_mnist
│   ├── assembled_surrogates.tar
│   └── LICENSE
├── pd1-data
│   ├── data.tar.gz
│   └── surrogates
│       ├── cifar100-wide_resnet-2048-train_cost.json
│       ├── cifar100-wide_resnet-2048-valid_error_rate.json
│       ├── imagenet-resnet-512-train_cost.json
│       ├── imagenet-resnet-512-valid_error_rate.json
│       ├── lm1b-transformer-2048-train_cost.json
│       ├── lm1b-transformer-2048-valid_error_rate.json
│       ├── surrogates.zip
│       ├── translate_wmt-xformer_translate-64-train_cost.json
│       └── translate_wmt-xformer_translate-64-valid_error_rate.json
└── yahpo-gym-data
    ├── benchmark_suites
    │   └── v1.0
    ├── fcnet
    ├── iaml_glmnet
    ├── iaml_ranger
    ├── iaml_rpart
    ├── iaml_super
    ├── iaml_xgboost
    ├── lcbench
    ├── nb301
    ├── rbv2_aknn
    ├── rbv2_glmnet
    ├── rbv2_ranger
    ├── rbv2_rpart
    ├── rbv2_super
    ├── rbv2_svm
    └── rbv2_xgboost
```

Once you have done so, you should be able to load the surrogates.

## Usage

### Running a singular local experiment

```bash
# We need at least two seeds for the plotting to work
just run random_search lm1b_transformer_2048_prior-at25 test 0
just run random_search lm1b_transformer_2048_prior-at25 test 1
just run priorband lm1b_transformer_2048_prior-at25 test 0
just run priorband lm1b_transformer_2048_prior-at25 test 1

# For more options see which can be run directly
python -m mf_prior_experiments.run -h
python -m mf_prior_experiments.run algorithm=random_search benchmark=lm1b_transformer_2048_prior-at25 experiment_group=test
```

### Running a grid of experiments on slurm

To run 5 seeds for two algorithms and benchmarks, e.g.,

```bash
just submit random_search,priorband lcbench189906_prior-at25 "range(0, 5)" test
# note the whitespace in `"range(0, 5)"`.

# Use `just submit` to see the arguments
```

### To get a list of benchmarks
```bash
just benchmarks
```

### To get a list of algorithms
```bash
just algorithms
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
  \"Good prior\": [\"lm1b_transformer_2048_prior-at25\"]
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
	--x_range_it 1 12
```
