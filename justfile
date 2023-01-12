# List available receipes
@list:
  just --list
# Run local experiment:
@run algorithm="random_search" benchmark="mfh3_bad" experiment_group="debug" seed="200" n_workers="1" :
  HYDRA_FULL_ERROR=1 python -m mf_prior_experiments.run \
    algorithm={{algorithm}} \
    benchmark={{benchmark}} \
    experiment_group={{experiment_group}} \
    seed={{seed}} \
    n_workers={{n_workers}}

# Submit job
@submit algorithms benchmarks seeds="range(1)" experiment_group="test" job_name="default" partition="mldlc_gpu-rtx2080" max_tasks="1000" time="0-23:59" memory="0" n_worker="1":
  python -m mf_prior_experiments.submit \
    --experiment_group {{experiment_group}} \
    --max_tasks {{max_tasks}} \
    --time {{time}} \
    --job_name {{job_name}} \
    --partition {{partition}} \
    --memory {{memory}} \
    --n_worker {{n_worker}} \
    --arguments algorithm={{algorithms}} benchmark={{benchmarks}} n_workers={{n_worker}} seed="{{seeds}}" hydra/job_logging=only_file \
    --exclude "kisexe20,kisexe28,kisexe34"

# Plot job
@plot experiment_group benchmarks algorithms filename ext="pdf" base_path=justfile_directory() :
  python -m mf_prior_experiments.plot \
    --experiment_group {{experiment_group}} \
    --benchmark {{benchmarks}} \
    --algorithm {{algorithms}} \
    --filename {{filename}} \
    --base_path {{base_path}} \
    --ext {{ext}} \
    --x_range 0 20 \
    --plot_default

# Table job
@table experiment_group benchmarks algorithms filename budget base_path=justfile_directory() :
  python -m mf_prior_experiments.final_table \
    --experiment_group {{experiment_group}} \
    --benchmark {{benchmarks}} \
    --algorithm {{algorithms}} \
    --filename {{filename}} \
    --base_path {{base_path}} \
    --budget {{budget}}

# List all available benchmarks
@benchmarks:
    ls -1 ./src/mf_prior_experiments/configs/benchmark | grep ".yaml" | sed -e "s/\.yaml$//"

# Generate all available benchmarks
@generate_benchmarks:
    python "./src/mf_prior_experiments/configs/benchmark/generate.py"

# List all available algorithms
@algorithms:
    ls -1 ./src/mf_prior_experiments/configs/algorithm | grep ".yaml" | sed -e "s/\.yaml$//"

@download:
    python -m mfpbench.download --data-dir "/work/dlclarge1/mallik-mf-prior/mf-prior-exp/data"

# This will run some compute on the login node, not ideal but it's fairly lightweight
@generate_priors:
    echo "I hope you know what you're doing"
    python "/work/dlclarge1/mallik-mf-prior/mf-prior-exp/src/mf-prior-bench/generate_priors.py" \
        --only "lm1b" "uniref50" "translatewmt" \
        --to "/work/dlclarge1/mallik-mf-prior/mf-prior-exp/src/mf-prior-bench/priors"
