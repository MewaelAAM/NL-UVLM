#!/bin/bash
#SBATCH -n 1
#SBATCH -c 35
#SBATCH --job-name=Bambula
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH -A kuin0119
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=neweafe2000@gmail.com
# SBATCH --exclusive
#SBATCH --nodelist=gpu-10-2
# module load miniconda/rig
# conda activate rig
echo "Finally - out of queue" 
echo "Running on $(hostname)"
nvidia-smi

# /home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/ft_s2.sh
python -c "import time; time.sleep(3*86400)"