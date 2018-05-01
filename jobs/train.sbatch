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
source activate l3embedding-cpu

SRCDIR=$HOME/dev/TempoEstimation

module purge

python $SRCDIR/train.py /beegfs/jtc440/aca/Beat_Tracking/hainsworth/samples /beegfs/jtc440/aca/Beat_Tracking/hainsworth/info hainsworth /scratch/jtc440/beat_tracker_output
