import json
import os
import numpy as np
from argparse import ArgumentParser
from ballroom import prep_ballroom_data
from data_utils import load_data
from hainsworth import prep_hainsworth_data
from evaluation import perform_evaluation
from log import init_console_logger
import logging

LOGGER = logging.getLogger('tempo_estimation')
LOGGER.setLevel(logging.DEBUG)

HOP_SIZE =  0.01

def parse_arguments():
    """
    Get command line arguments
    """
    parser = ArgumentParser(description='Evaluate a deep beat tracker model')
    parser.add_argument('model_dir', help='Path to model directory')

    # Creates a namespace object
    args = parser.parse_args()
    # vars creates a dictionary where key:value pairs attribute name:attribute
    return vars(args)


# CHANGE SAMPLE RATE TO 24kHz or 16kHz
def main(model_dir):
    """
    Train a deep beat tracker model
    """
    # Set up logger
    init_console_logger(LOGGER, verbose=True)

    config_path = os.path.join(model_dir, 'config.json')
    feature_data_dir = os.path.join(model_dir, '../../data')

    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = config['data_dir']
    label_dir = config['label_dir']
    dataset = config['dataset']
    output_dir = config['output_dir']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    lr = config['lr']
    target_fs = config['target_fs']
    audio_window_size = config['audio_window_size']
    model_type = config['model_type']
    k_smoothing = config.get('k_smoothing', 1)
    hop_length = int(target_fs * HOP_SIZE)

    LOGGER.info('Loading {} data.'.format(dataset))
    if dataset == 'hainsworth':
        a, r = prep_hainsworth_data(data_dir, label_dir, target_fs,
                                    load_audio=False)
    elif dataset == 'ballroom':
        a, r = prep_ballroom_data(data_dir, label_dir, hop_length, target_fs,
                                  load_audio=False)

    train_data_path = os.path.join(feature_data_dir, '{}_train_data.npz').format(dataset)
    valid_data_path = os.path.join(feature_data_dir, '{}_valid_data.npz').format(dataset)
    test_data_path = os.path.join(feature_data_dir, '{}_test_data.npz').format(dataset)

    # Otherwise, just load existing data
    train_data = load_data(train_data_path, model_type)
    valid_data = load_data(valid_data_path, model_type)
    test_data = load_data(test_data_path, model_type)

    # Evaluate model
    LOGGER.info('Evaluating model.')
    perform_evaluation(train_data, valid_data, test_data, model_dir, r,
                       target_fs, batch_size, k_smoothing=k_smoothing)

    LOGGER.info('Done!')


if __name__ == '__main__':
    main(**parse_arguments())
