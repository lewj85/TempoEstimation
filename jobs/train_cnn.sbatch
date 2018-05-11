#!/bin/bash

# Jason Cramer, Jesse Lew

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
source activate tempoenv

SRCDIR=$HOME/audio/TempoEstimation

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/train.py \
    --num-epochs 200 \
    --batch-size 5 \
    --lr 0.001 \
    --model audio \
    --target-fs 16000 \
    --audio-window-size 512 \
    $SRCDIR/data_config.json \
    /scratch/jtc440/beat_tracker_output
