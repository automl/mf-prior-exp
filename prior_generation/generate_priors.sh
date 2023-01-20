#!/bin/bash
#SBATCH --export=ALL
#SBATCH --cpus-per-task=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=32G
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --array 0-20
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
# 50,000 (1mill would take wayyy too long)
# 50,000 is about 30 mins on the slowest at max
NSAMPLES=50000


GOOD_NAME="good"
GOOD_INDEX="0"
GOOD_EPSILON="0.01"
GOOD_CATEGORICAL_CHANGE_CHANCE="None"

GOOD_PRIOR="${GOOD_NAME}:${GOOD_INDEX}:${GOOD_EPSILON}:${GOOD_CATEGORICAL_CHANGE_CHANCE}"

MEDIUM_NAME="medium"
MEDIUM_INDEX="0"
MEDIUM_EPSILON="0.25"
MEDIUM_CATEGORICAL_CHANGE_CHANCE="None"

MEDIUM_PRIOR="${MEDIUM_NAME}:${MEDIUM_INDEX}:${MEDIUM_EPSILON}:${MEDIUM_CATEGORICAL_CHANGE_CHANCE}"

BAD_NAME="bad"
BAD_INDEX="-1"
BAD_EPSILON="None"
BAD_CATEGORICAL_CHANGE_CHANCE="None"

BAD_PRIOR="${BAD_NAME}:${BAD_INDEX}:${BAD_EPSILON}:${BAD_CATEGORICAL_CHANGE_CHANCE}"


ARGS=(
    "lcbench-126026"
    "lcbench-167190"
    "lcbench-168330"
    "lcbench-168910"
    "lcbench-189906"
    "jahs_cifar10"
    "jahs_colorectal_histology"
    "jahs_fashion_mnist"
    "mfh3_terrible"
    "mfh3_bad"
    "mfh3_moderate"
    "mfh3_good"
    "mfh6_terrible"
    "mfh6_bad"
    "mfh6_moderate"
    "mfh6_good"
    "lm1b_transformer_2048"
    "uniref50_transformer_128"
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
    python -m mfpbench generate-priors \
        --to "${PRIORDIR}" \
        --seed "${SEED}" \
        --nsamples "${NSAMPLES}" \
        --only "${selection}" \
        --priors "${GOOD_PRIOR}" "${MEDIUM_PRIOR}" "${BAD_PRIOR}" \
        --use-hartmann-optimum "${GOOD_NAME}" "${MEDIUM_NAME}"
else
    python -m mfpbench generate-priors \
        --to "${PRIORDIR}" \
        --seed "${SEED}" \
        --nsamples "${NSAMPLES}" \
        --only "${selection}" \
        --priors "${GOOD_PRIOR}" "${MEDIUM_PRIOR}" "${BAD_PRIOR}"
fi
