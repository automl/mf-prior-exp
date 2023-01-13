#!/bin/bash
#SBATCH --export=ALL
#SBATCH --cpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --mem=25000
#SBATCH --partition=bosch_cpu-cascadelake

#SBATCH --job-name=priors-mfpbench
#
if [[ -z "${SLURM_JOB_USER}" ]]; then
    echo "Use sbatch to call this file"
    exit
fi

prior_directory="$(pwd)/src/mf-prior-bench/priors"
echo "Running:"
python -m mfpbench generate-priors \
    --to "${prior_directory}" \
    --seed 1330777 \
    --nsamples 100 \
    --clean
