
#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=modl_demo
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

python -m pip install cupy