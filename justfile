# List available receipes
@list:
  just --list

# Run local experiment:
@run algorithm="random_search" benchmark="jahs_cifar10":
  python -m mf_prior_experiments.run \
    algorithm={{algorithm}} \
    benchmark={{benchmark}}


# Submit job
@submit algorithms benchmarks seeds="range(1)" experiment_group="test" job_name="default" partition="mldlc_gpu-rtx2080" max_tasks="1000" time="0-23:59" memory="0":
  python -m mf_prior_experiments.submit \
    --experiment_group {{experiment_group}} \
    --max_tasks {{max_tasks}} \
    --time {{time}} \
    --job_name {{job_name}} \
    --partition {{partition}} \
    --memory {{memory}} \
    --arguments algorithm={{algorithms}} benchmark={{benchmarks}} seed="{{seeds}}" hydra/job_logging=only_file
