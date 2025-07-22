#!/bin/bash
#BSUB -o sm_logs/snakemake_master-%J-output.log
#BSUB -e sm_logs/snakemake_master-%J-error.log 
#BSUB -q oversubscribed
#BSUB -G team152
#BSUB -n 1
#BSUB -M 10000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>10000] rusage[mem=10000] span[hosts=1]"
#BSUB -J 1

# Define some params
config_var=configs/config.yaml
worfklow_prefix="R1_"
group="team152"
workdir=${PWD}

# Load snakemake and singulatiry
module load HGI/common/snakemake/7
module load ISG/singularity/3.11.4
which singularity

# Make a log dir
mkdir -p sm_logs

#conda activate scvi-env
# module load HGI/softpack/groups/hgi/snakemake/7.32.4
#module load ISG/singularity/3.9.0 # farm5
module load ISG/singularity/3.11.4 # farm22

# For the CUDA libraries for GPU
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/software/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spac/linux-ubuntu22.04-x86_64_v3/gcc-11.4.0/cuda-11.8.0-vw5pdvpi33mowuxd7w3zgmrbfejiqxqb/targets/x86_64-linux/lib

# Copy config to results
mkdir -p results
cp $config_var results/

# Build dag
snakemake -j 20000 \
    --latency-wait 90 \
    --use-envmodules \
    --rerun-incomplete \
    --keep-going \
    --directory ${workdir} \
    --configfile ${config_var} \
    --use-singularity \
    --singularity-args "-B /lustre,/software,/nfs/users/nfs_b/bh18/.local/lib/python3.7/site-packages" \
    --keep-going \
    --restart-times 0 \
    --snakefile workflows/Snakefile \
    --dag | dot -Tpng > dags/dag.png

# Execute script (updating config params to use optimum model params)
snakemake -j 20000 \
    --latency-wait 90 \
    --use-envmodules \
    --rerun-incomplete \
    --keep-going \
    --directory ${workdir} \
    --configfile ${config_var} \
    --use-singularity \
    --singularity-args "-B /lustre,/software,/nfs/users/nfs_b/bh18/.local/lib/python3.7/site-packages --nv" \
    --keep-going \
    --restart-times 0 \
    --snakefile workflows/Snakefile

# NOTE: Have adjusted to run originalk model to test
# Add the following to overwrite with optimum params
# --config optimise_run_params=False sparsity_l1__activity=0.01 sparsity_l1__bias=0.0001 sparsity_l1__kernel=0.0001 sparsity_l2__activity=0.0001 sparsity_l2__bias=0.01 sparsity_l2__kernel=0.01 \


# bsub -M 2000 -a "memlimit=True" -R "select[mem>2000] rusage[mem=2000] span[hosts=1]" -o sm_logs/snakemake_master-%J-output.log -e sm_logs/snakemake_master-%J-error.log -q oversubscribed -J "snakemake_master_R1" < submit_snakemake.sh 
