# PriorBand: Hyperband + Human Expert Knowledge

This branch of the repo contains the code and setup for *reproducing experiments* in [the paper](https://openreview.net/forum?id=ds21dwfBBH) published at the *6th Workshop on Meta-Learning* at **NeurIPS 2022**.

To cite the paper or code:
```bibtex
@inproceedings{mallik2022priorband,
  title={PriorBand: HyperBand + Human Expert Knowledge},
  author={N. Mallik and C. Hvarfner and D. Stoll and M. Janowski and E. Bergman and M. Lindauer and L. Nardi and F. Hutter},
  booktitle={Sixth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=ds21dwfBBH}
}
```

## Installation

### 1. Clone the repository

```bash
git clone --recursive https://github.com/automl/mf-prior-exp.git

# To use the version used for the paper
git clone --branch "vPaper-priorband" --recursive https://github.com/automl/mf-prior-exp.git

# If you forgot to use the --recursive flag or checkout the branches, you can do the following
# to manuall perform the steps
cd mf-prior-exp
git checkout vPaper-priorband
git submodule update --init
```

### 2. Conda, Poetry, Package, Pre-Commit

To setup tooling and install the package, follow [this documentation](https://automl.github.io/neps/0.5.1/contributing/installation/) using the environment name of your choice.

```bash
poetry install
```

**NOTE**: The version used for the paper `vPaper-PriorBand` requires an older version of `NePS` which requires Python version \<3.8 so consider creating an environment with `conda create -n mf-prior python=3.7.12`

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
python -m mfpbench.download --data-dir data
```

## Usage

### Running a singular local experiment

```bash
just run random_search jahs_cifar10
```

For more options see

```bash
python -m mf_prior_experiments.run -h
```

and run the python command directly, e.g.,

```bash
python -m mf_prior_experiments.run algorithm=random_search benchmark=jahs_cifar10 experiment_group=debug
```

### Running a grid of experiments on slurm

To run 10 seeds for two algorithms and benchmarks, e.g.,

```
just submit alg_a,alg_b bench_a,bench_b "range(0, 10)" 22-08-18_updated-priors
```

note the whitespace in `"range(0, 10)"`.

For more options see

```
just
```

### Analysing experiments

## Contributing

### Working with git submodules

See [the git-scm documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules). In short:

To pull in changes for `mf-prior-exp` and all submodules (`neps` and `mf-prior-bench`) run

```bash
git pull --recurse-submodules
```

To pull in changes from one submodule, change to its directory and run

```bash
git fetch
git merge origin/main
```

To code in the submodule first change to its directory, then checkout the branch you want to work on, e.g.,

```bash
cd src/neps
git checkout master
```

then perform, commit, and push your changes to the submodule's repository

```bash
pre-commit install
git commit -m "awesome changes"
git push
```

next, you also need to commit the changed submodule, e.g.,

```bash
cd ../..
git add src/neps
git commit -m "update neps submodule"
git push
```

### Managing dependencies

For how to manage dependencies see [the overview on poetry](https://automl.github.io/neps/0.5.1/contributing/dependencies/).

### Tooling

There is also some [documentation for the tools](https://automl.github.io/neps/0.5.1/contributing/tests/#disabling-and-skipping-checks-etc) in this repo.

