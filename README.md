# MF + Priors

## Installation

### Conda, Poetry, Package, Pre-Commit

Clone the repository and follow [this documentation](https://automl.github.io/neps/contributing/installation/) using the environment name of your choice.

### Just

To install just you can check the [just documentation](https://github.com/casey/just#installation), or run the below command

```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to $HOME/.just
```

Also, to make the `just` command available you should add

```bash
export PATH="$HOME/.just:$PATH"
```

to your `.zshrc` / `.bashrc` or alternatively simply run the export manually.

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

### Managing dependencies

For how to manage dependencies see [the overview on poetry](https://automl.github.io/neps/contributing/dependencies/).

### Tooling

There is also some [documentation for the tools](https://automl.github.io/neps/contributing/faq/) in this repo.
