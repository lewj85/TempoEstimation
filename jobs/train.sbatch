#!/bin/bash
#
#SBATCH --job-name=Tempo_GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=name@email.org
#SBATCH --output="train-beat-tracker-%j.out"
#SBATCH --err="train-beat-tracker-%j.err"

source ~/.bashrc
cd /home/$USER/dev
source activate tempo

SRCDIR=$HOME/dev/TempoEstimation

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/train.py \
    --num-epochs 200 \
    --batch-size 10 \
    --lr 0.001 \
    $SRCDIR/data_config.json \
    /scratch/jtc440/beat_tracker_output
