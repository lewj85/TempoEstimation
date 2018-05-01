import os
import numpy as np
import pickle as pk
from argparse import ArgumentParser
from hainsworth import prep_hainsworth_data
from preprocess_data import preprocess_data
from model.architectures import construct_spectrogram_bilstm
from model.training import train_model
from evaluation import compute_beat_metrics, compute_tempo_metrics
from data_utils import create_data_subsets
from beat_tracking import estimate_beats_for_batch, \
    get_beat_times_from_annotations
from tempo import get_tempos_from_annotations, estimate_tempos_for_batch
from math import ceil
from keras.models import load_model
from log import init_console_logger
import logging

LOGGER = logging.getLogger('tempo_estimation')
LOGGER.setLevel(logging.DEBUG)

HOP_SIZE = 441
TARGET_FS = 44100

HAINSWORTH_MIN_TEMPO = 40
HAINSWORTH_MAX_TEMPO = 250
HAINSWORTH_MIN_LAG = int(60 * TARGET_FS / (HOP_SIZE * HAINSWORTH_MAX_TEMPO))
HAINSWORTH_MAX_LAG = ceil(60 * TARGET_FS / (HOP_SIZE * HAINSWORTH_MIN_TEMPO))

def parse_arguments():
    """
    Get command line arguments
    """
    parser = ArgumentParser(description='Train a deep beat tracker model')
    parser.add_argument('data_dir', help='Path to audio directory')
    parser.add_argument('label_dir', help='Path to annotation directory')
    parser.add_argument('dataset', choices=['hainsworth', 'ballroom'],
                        help='Name of dataset')
    parser.add_argument('output_dir', help='Path where output will be saved')
    # -m is shortcut for --model
    parser.add_argument('--model', '-m', dest='model_type', default='spectrogram',
                        choices=['spectrogram', 'audio'], help='Model type')
    # TODO: Add command line arguments for training hyperparameters


    # Creates a namespace object
    args = parser.parse_args()
    # vars creates a dictionary where key:value pairs attribute name:attribute
    return vars(args)


def main(data_dir, label_dir, dataset, output_dir, model_type='spectrogram'):
    """
    Train a deep beat tracker model
    """
    # Set up logger
    init_console_logger(LOGGER, verbose=True)

    LOGGER.info('Loading {} data.'.format(dataset))
    train_data_path = os.path.join(output_dir, '{}_train_data.pkl').format(dataset)
    valid_data_path = os.path.join(output_dir, '{}_valid_data.pkl').format(dataset)
    test_data_path = os.path.join(output_dir, '{}_test_data.pkl').format(dataset)

    data_exists = os.path.exists(train_data_path) \
        and os.path.exists(train_data_path) \
        and os.path.exists(train_data_path)

    # Load audio and annotations
    if dataset == 'hainsworth':
        a, r = prep_hainsworth_data(data_dir, label_dir, TARGET_FS)
    elif dataset == 'ballroom':
        a, r = prep_ballroom_data(data_dir, label_dir, HOP_SIZE, TARGET_FS)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not data_exists:
        # Create preprocessed data if it doesn't exist
        LOGGER.info('Preprocessing data for model type "{}".'.format(model_type))
        # Get features and targets from data
        X, y = preprocess_data(a, r, mode=model_type)

        LOGGER.info('Creating data subsets.')
        train_data, valid_data, test_data = create_data_subsets(X, y)

        LOGGER.info('Saving data subsets to disk.')
        with open(train_data_path, 'wb') as f:
            pk.dump(train_data, f)
        with open(valid_data_path, 'wb') as f:
            pk.dump(valid_data, f)
        with open(test_data_path, 'wb') as f:
            pk.dump(test_data, f)

    else:
        # Otherwise, just load existing data
        with open(train_data_path, 'rb') as f:
            train_data = pk.load(f)
        with open(valid_data_path, 'rb') as f:
            valid_data = pk.load(f)
        with open(test_data_path, 'rb') as f:
            test_data = pk.load(f)

    # TODO: Change model name based on dataset
    model_path = os.path.join(output_dir, 'model.hdf5')
    if not os.path.exists(model_path):
        # Only train model if we haven't done so already
        LOGGER.info('Training model.')
        # Create, train, and save model
        model_path = train_model(train_data, valid_data, model_type, output_dir,
                                 lr=0.0001, batch_size=5, num_epochs=10)

    LOGGER.info('Loading model.')
    model = load_model(model_path)

    frame_rate = TARGET_FS / HOP_SIZE

    LOGGER.info('Running model on data.')
    y_train_pred = model.predict(train_data['X'])[:,:,1]
    y_valid_pred = model.predict(valid_data['X'])[:,:,1]
    y_test_pred = model.predict(test_data['X'])[:,:,1]

    # Save model outputs
    outputs = {
        'train': y_train_pred,
        'valid': y_valid_pred,
        'test': y_test_pred,
    }
    output_path = os.path.join(output_dir, 'output.npz')
    LOGGER.info('Saving model outputs.')
    np.savez(output_path, **outputs)

    # Using test data, estimate beats and evaluate
    LOGGER.info('Estimating beats.')
    beat_times_train = get_beat_times_from_annotations(r, train_data['indices'])
    beat_times_valid = get_beat_times_from_annotations(r, valid_data['indices'])
    beat_times_test = get_beat_times_from_annotations(r, test_data['indices'])

    beat_times_pred_train = estimate_beats_for_batch(y_train_pred, frame_rate,
        HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG)
    beat_times_pred_valid = estimate_beats_for_batch(y_valid_pred, frame_rate,
        HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG)
    beat_times_pred_test = estimate_beats_for_batch(y_test_pred, frame_rate,
        HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG)

    LOGGER.info('Computing beat tracking metrics.')
    beat_metrics_train = compute_beat_metrics(beat_times_train, beat_times_pred_train)
    beat_metrics_valid = compute_beat_metrics(beat_times_valid, beat_times_pred_valid)
    beat_metrics_test = compute_beat_metrics(beat_times_test, beat_times_pred_test)

    beat_metrics = {
        'train': beat_metrics_train,
        'valid': beat_metrics_valid,
        'test': beat_metrics_test,
    }

    beat_metrics_path = os.path.join(output_dir, 'beat_metrics.pkl')
    LOGGER.info('Saving beat tracking metrics.')
    with open(beat_metrics_path, 'wb') as f:
        pk.dump(beat_metrics, f)

    # Using test data, estimate tempo and evaluate
    LOGGER.info('Estimating tempo.')
    tempos_train = get_tempos_from_annotations(r, train_data['indices'])
    tempos_valid = get_tempos_from_annotations(r, valid_data['indices'])
    tempos_test = get_tempos_from_annotations(r, test_data['indices'])

    tempos_pred_train = estimate_tempos_for_batch(y_train_pred, frame_rate,
                                 HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14)
    tempos_pred_valid = estimate_tempos_for_batch(y_valid_pred, frame_rate,
                                 HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14)
    tempos_pred_test = estimate_tempos_for_batch(y_test_pred, frame_rate,
                                 HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14)

    LOGGER.info('Computing tempo estimation metrics.')
    tempo_metrics_train = compute_tempo_metrics(tempos_train, tempos_pred_train)
    tempo_metrics_valid = compute_tempo_metrics(tempos_valid, tempos_pred_valid)
    tempo_metrics_test = compute_tempo_metrics(tempos_test, tempos_pred_test)

    tempo_metrics = {
        'train': tempo_metrics_train,
        'valid': tempo_metrics_valid,
        'test': tempo_metrics_test,
    }

    LOGGER.info('Saving tempo estimation metrics.')
    tempo_metrics_path = os.path.join(output_dir, 'tempo_metrics.pkl')
    with open(tempo_metrics_path, 'wb') as f:
        pk.dump(tempo_metrics, f)

    LOGGER.info('Done!')


if __name__ == '__main__':
    main(**parse_arguments())
