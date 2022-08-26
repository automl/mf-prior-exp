# MF + Priors

## Installation

### 1. Clone the repository

```bash
git clone --recursive https://github.com/automl/mf-prior-exp.git
```

### 2. Conda, Poetry, Package, Pre-Commit

To setup tooling, follow [this documentation](https://automl.github.io/neps/contributing/installation/) using the environment name of your choice.

**NOTE**: NePS requires Python version \<3.8 so consider creating an environment with `conda create -n mf-prior python=3.7.12`

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
python -m mfpbench.download --data-dir data`
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
just submit alg_a,alg_b bench_a,bench_b "range(0,10)" 22-08-18_updated-priors
```

for more options see

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

also make sure to use pre-commit

```bash
pre-commit install
```

then perform and commit your changes to the submodule's repository. Next, you also need to commit the changed submodule, e.g.,

```bash
cd ../..

```


### Managing dependencies

For how to manage dependencies see [the overview on poetry](https://automl.github.io/neps/contributing/dependencies/).

### Tooling

There is also some [documentation for the tools](https://automl.github.io/neps/contributing/faq/) in this repo.
