#!/bin/bash
#SBATCH --export=ALL
#SBATCH --cpus-per-task=1
#SBATCH --time=0-03:00:00
#SBATCH --mem=30G
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --array 0-4
#SBATCH --output logs/%a.log
#SBATCH --error logs/%a.log
#SBATCH --job-name=priors-mfpbench

WORKDIR="/work/dlclarge1/mallik-mf-prior/mf-prior-exp"
PRIORDIR="${WORKDIR}/src/mf-prior-bench/priors"

if [[ -z "${SLURM_JOB_USER}" ]]; then
    echo "Use sbatch to call this file"
    exit
fi

SEED=133077
FIFTY_K=50000
AT_25=25

# NOTE: If editing this, make sure to change the array job thing above
ARGS=(
    "lcbench-189906"
    "lm1b_transformer_2048"
    "translatewmt_xformer_64"
    "cifar100_wideresnet_2048"
    "imagenet_resnet_512"
)

selection=${ARGS[$SLURM_ARRAY_TASK_ID]}

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Running: ${selection}"
echo "Running from: ${WORKDIR}"

cd $WORKDIR

if [[ $selection == "mfh"* ]]; then

    # [Good]
    # * good:0:0:None will just get replaced by the optimum and we will add noise
    #   in the benchmark yamls.
    #   This optimum is done by `--use-hartmann-optimum good` which replaces the
    #   defininition of "good" below with the optimum configuration.
    #   This is why we only have to use 1 sample as this sample will get replaced
    #   by the optimum
    #python -m mfpbench generate-priors \
        #--to "${PRIORDIR}" \
        #--seed "${SEED}" \
        #--nsamples "1" \
        #--only "${selection}" \
        #--priors "good:0:0:None" \
        #--use-hartmann-optimum "good"

    # [at25]
    # * Random search at 25 samples and take the best one, no noise
    #python -m mfpbench generate-priors \
        #--to "${PRIORDIR}" \
        #--seed "${SEED}" \
        #--nsamples "${AT_25}" \
        #--only "${selection}" \
        #--priors "at25:0:0:None" 

    # [Bad]
    # * Random search at 50k samples and take the worst
    #python -m mfpbench generate-priors \
    #    --to "${PRIORDIR}" \
    #    --seed "${SEED}" \
    #    --nsamples "${FIFTY_K}" \
    #    --only "${selection}" \
    #    --priors "bad:-1:0:None" 
    echo "mfh"
else

    # [Good]
    # * Find to optimum of 50k samples and perturb the config by 0.01
    python -m mfpbench generate-priors \
        --to "${PRIORDIR}" \
        --seed "${SEED}" \
        --nsamples "${FIFTY_K}" \
        --only "${selection}" \
        --priors "good:0:0.01:None"

    # [Medium]
    # * Find the optimum of 50k samples
    # * Noise will be added later in the benchmarks with
    #   `perturb_prior=.250`
#    python -m mfpbench generate-priors \
#        --to "${PRIORDIR}" \
#        --seed "${SEED}" \
#        --nsamples "${FIFTY_K}" \
#        --only "${selection}" \
#        --priors "medium:0:0:None"
#
    # [at25]
    # * Find the optimum of 25 random samples, no noise
#    python -m mfpbench generate-priors \
#        --to "${PRIORDIR}" \
#        --seed "${SEED}" \
#        --nsamples "${AT_25}" \
#        --only "${selection}" \
#        --priors "at25:0:0:None"

    # [Bad]
    # * Find the worst of 50k samples, no noise
#    python -m mfpbench generate-priors \
#        --to "${PRIORDIR}" \
#        --seed "${SEED}" \
#        --nsamples "${FIFTY_K}" \
#        --only "${selection}" \
#        --priors "bad:-1:0:None"
fi
