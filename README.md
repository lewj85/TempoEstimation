# Beat Tracking and Tempo Estimation using Learned Convolutional Features and Tempo Weighting

This repository contains code for training and evaluating a beat tracker and tempo estimation model using neural networks.

`train.py` can be used to train a model. The usage can be seen by running `python train.py -h`. The format for the data configuration file (one of the expected inputs) is exemplified in `data_config.json`. This file describes the locations of the audio and annotation directories for the datasets used in the training set and the testing set.

`evaluate.py` can be used for evaluating a trained model on beat tracking and tempo estimation. Note that `train.py` already runs the evaluation after training.

