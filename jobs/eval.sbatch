#!/bin/bash

# Jason Cramer, Jesse Lew

#SBATCH --job-name=Tempo_GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=name@email.org
#SBATCH --output="eval-beat-tracker-%j.out"
#SBATCH --err="eval-beat-tracker-%j.err"

source ~/.bashrc
cd /home/$USER/dev
source activate tempoenv

SRCDIR=$HOME/audio/TempoEstimation

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/evaluate.py /scratch/jtc440/beat_tracker_output/spectrogram/hainsworth/model/20180504235042
