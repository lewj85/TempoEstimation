import datetime
import json
import os
import numpy as np
from argparse import ArgumentParser
from ballroom import prep_ballroom_data
from hainsworth import prep_hainsworth_data
from preprocess_data import preprocess_data
from model.training import train_model
from evaluation import perform_evaluation
from data_utils import create_data_subsets, load_data
from math import ceil
from log import init_console_logger
import logging

LOGGER = logging.getLogger('tempo_estimation')
LOGGER.setLevel(logging.DEBUG)

HOP_SIZE =  0.01

def parse_arguments():
    """
    Get command line arguments
    """
    parser = ArgumentParser(description='Train a deep beat tracker model')
    parser.add_argument('data_config', help='Path to data config file')
    parser.add_argument('output_dir', help='Path where output will be saved')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs for training model')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Batch size for training')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--target-fs', type=int, default=44100,
                        help='Target sample rate. If spectrogram samples is used, must be 44100')
    parser.add_argument('--audio-window-size', type=int, default=2048,
                        help='Audio window size')
    parser.add_argument('--k-smoothing', type=int, default=1,
                        help='k for k-smothing')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    # -m is shortcut for --model
    parser.add_argument('--model', '-m', dest='model_type', default='spectrogram',
                        choices=['spectrogram', 'audio'], help='Model type')
    # TODO: Add command line arguments for training hyperparameters


    # Creates a namespace object
    args = parser.parse_args()
    # vars creates a dictionary where key:value pairs attribute name:attribute
    return vars(args)


# CHANGE SAMPLE RATE TO 24kHz or 16kHz
def main(data_config, output_dir, num_epochs=10, batch_size=5,
         lr=0.001, target_fs=44100, audio_window_size=2048, patience=5,
         model_type='spectrogram', k_smoothing=1):
    """
    Train a deep beat tracker model
    """
    # Set up logger
    init_console_logger(LOGGER, verbose=True)

    with open(data_config, 'r') as f:
        data_config = json.load(f)

    sorted_train_datasets = sorted(data_config['train'].keys())
    train_dataset_desc = "train_" + "_".join(sorted_train_datasets)
    test_dataset_desc = "test_" + "_".join(sorted(data_config['test'].keys()))

    dataset_desc = train_dataset_desc + "-" + test_dataset_desc

    output_dir = os.path.join(output_dir, model_type, dataset_desc)
    LOGGER.info('Output will be saved to {}'.format(output_dir))

    feature_data_dir = os.path.join(output_dir, 'data')
    model_dir = os.path.join(output_dir, 'model',
                             datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(feature_data_dir):
        os.makedirs(feature_data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    LOGGER.info('Saving configuration.')
    config = {
        'data_config': data_config,
        'output_dir': output_dir,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'patience': patience,
        'k_smoothing': k_smoothing,
        'target_fs': target_fs,
        'audio_window_size': audio_window_size,
        'model_type': model_type
    }

    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)

    LOGGER.info('Loading {} data.'.format(dataset_desc))
    train_data_path = os.path.join(feature_data_dir, '{}_train_data.npz').format(dataset_desc)
    valid_data_path = os.path.join(feature_data_dir, '{}_valid_data.npz').format(dataset_desc)
    test_data_path = os.path.join(feature_data_dir, '{}_test_data.npz').format(dataset_desc)


    data_exists = os.path.exists(train_data_path) \
        and os.path.exists(valid_data_path) \
        and os.path.exists(test_data_path)

    if model_type == 'spectrogram':
        assert target_fs == 44100

    hop_length = int(target_fs * HOP_SIZE)

    sorted_train_datasets = sorted(data_config['train'].keys())
    a_train = []
    r_train = []
    # Load audio and annotations
    for dataset in sorted_train_datasets:
        data_dir = data_config['train'][dataset]['data_dir']
        label_dir = data_config['train'][dataset]['label_dir']
        if dataset == 'hainsworth':
            a, r = prep_hainsworth_data(data_dir, label_dir, target_fs,
                                        load_audio=not data_exists)
        elif dataset == 'ballroom':
            a, r = prep_ballroom_data(data_dir, label_dir, target_fs,
                                      load_audio=not data_exists)

        a_train += a
        r_train += r

    a_test = []
    r_test = []
    for dataset, dataset_dirs in data_config['test'].items():
        data_dir = dataset_dirs['data_dir']
        label_dir = dataset_dirs['label_dir']
        if dataset == 'hainsworth':
            a, r = prep_hainsworth_data(data_dir, label_dir, target_fs,
                                        load_audio=not data_exists)
        elif dataset == 'ballroom':
            a, r = prep_ballroom_data(data_dir, label_dir, target_fs,
                                      load_audio=not data_exists)

        a_test += a
        r_test += r

    if not data_exists:
        # Create preprocessed data if it doesn't exist
        LOGGER.info('Preprocessing data for model type "{}".'.format(model_type))
        # Get features and targets from data
        X_train, y_train = preprocess_data(a_train, r_train, mode=model_type,
            hop_size=hop_length, audio_window_size=audio_window_size, sr=target_fs)
        X_test, y_test = preprocess_data(a_test, r_test, mode=model_type,
            hop_size=hop_length, audio_window_size=audio_window_size, sr=target_fs)

        test_data = {
            'X': X_test,
            'y': y_test,
            'indices': np.arange(len(y_test)) # Hack
        }

        LOGGER.info('Creating data subsets.')
        train_data, valid_data = create_data_subsets(X_train, y_train)

        LOGGER.info('Saving data subsets to disk.')
        np.savez(train_data_path, **train_data)
        np.savez(valid_data_path, **valid_data)
        np.savez(test_data_path, **test_data)

    else:
        # Otherwise, just load existing data
        train_data = load_data(train_data_path, model_type)
        valid_data = load_data(valid_data_path, model_type)
        test_data = load_data(test_data_path, model_type)

    model_path = os.path.join(model_dir, 'model.hdf5')
    if not os.path.exists(model_path):
        # Only train model if we haven't done so already
        LOGGER.info('Training model.')
        # Create, train, and save model
        model_path = train_model(train_data, valid_data, model_type, model_path,
                                 lr=lr, batch_size=batch_size, num_epochs=num_epochs,
                                 audio_window_size=audio_window_size, patience=patience)

    # Evaluate model
    LOGGER.info('Evaluating model.')
    perform_evaluation(train_data, valid_data, test_data, model_dir, r_train, r_test,
                       target_fs, batch_size, k_smoothing=k_smoothing)

    LOGGER.info('Done!')


if __name__ == '__main__':
    main(**parse_arguments())
