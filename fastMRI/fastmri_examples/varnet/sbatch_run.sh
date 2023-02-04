#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=train_varnet_demo
#SBATCH --time=23:00:00
#SBATCH --mail-user=hernan.aguilera@fau.de
#SBATCH --mail-type=ALL
#
# do not export environment variables
#SBATCH --export=NONE



unset SLURM_EXPORT_ENV

module load cuda
module load python/3.8-anaconda
# conda init bash
eval "$(conda shell.bash hook)"
conda activate myenv

# mkdir $TMPDIR/$SLURM_JOB_ID

# cp -r /home/vault/iwbi/iwbi005h/fastMRI_data/knee/* $TMPDIR/$SLURM_JOB_ID

python train_varnet_demo.py --data_path /home/vault/iwbi/shared/cip22_varnet_modl/brain/ --mode "test" # "test" # "val"
# rm -r $TMPDIR/$SLURM_JOB_ID
