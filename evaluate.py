import json
import os
import numpy as np
import pickle as pk
from argparse import ArgumentParser
from ballroom import prep_ballroom_data
from hainsworth import prep_hainsworth_data
from evaluation import compute_beat_metrics, compute_tempo_metrics
from beat_tracking import estimate_beats_for_batch, \
    get_beat_times_from_annotations
from tempo import get_tempos_from_annotations, estimate_tempos_for_batch
from math import ceil
from keras.models import load_model
from log import init_console_logger
import logging

LOGGER = logging.getLogger('tempo_estimation')
LOGGER.setLevel(logging.DEBUG)

from train import HOP_SIZE, HAINSWORTH_MIN_TEMPO, HAINSWORTH_MAX_TEMPO

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
    feature_data_dir = os.path.join(output_dir, 'data')
    hop_length = int(target_fs * HOP_SIZE)

    LOGGER.info('Loading {} data.'.format(dataset))
    train_data_path = os.path.join(feature_data_dir, '{}_train_data.npz').format(dataset)
    valid_data_path = os.path.join(feature_data_dir, '{}_valid_data.npz').format(dataset)
    test_data_path = os.path.join(feature_data_dir, '{}_test_data.npz').format(dataset)

    if dataset == 'hainsworth':
        a, r = prep_hainsworth_data(data_dir, label_dir, target_fs,
                                    load_audio=False)
    elif dataset == 'ballroom':
        a, r = prep_ballroom_data(data_dir, label_dir, hop_length, target_fs,
                                  load_audio=False)

    LOGGER.info('Loading model.')
    model_path = os.path.join(model_dir, 'model.hdf5')
    model = load_model(model_path)

    frame_rate = target_fs / hop_length

    LOGGER.info('Running model on data.')
    y_train_pred = model.predict(train_data['X'], batch_size=batch_size)[:,:,1]
    y_valid_pred = model.predict(valid_data['X'], batch_size=batch_size)[:,:,1]
    y_test_pred = model.predict(test_data['X'], batch_size=batch_size)[:,:,1]

    # Save model outputs
    outputs = {
        'train': y_train_pred,
        'valid': y_valid_pred,
        'test': y_test_pred,
    }
    output_path = os.path.join(model_dir, 'output.npz')
    LOGGER.info('Saving model outputs.')
    np.savez(output_path, **outputs)


    min_lag = int(60 * target_fs / (hop_length * HAINSWORTH_MAX_TEMPO))
    max_lag = ceil(60 * target_fs / (hop_length * HAINSWORTH_MIN_TEMPO))

    # Using test data, estimate beats and evaluate
    LOGGER.info('Estimating beats.')
    beat_times_train = get_beat_times_from_annotations(r, train_data['indices'])
    beat_times_valid = get_beat_times_from_annotations(r, valid_data['indices'])
    beat_times_test = get_beat_times_from_annotations(r, test_data['indices'])

    beat_times_pred_train = estimate_beats_for_batch(y_train_pred, frame_rate,
        min_lag, max_lag)
    beat_times_pred_valid = estimate_beats_for_batch(y_valid_pred, frame_rate,
        min_lag, max_lag)
    beat_times_pred_test = estimate_beats_for_batch(y_test_pred, frame_rate,
        min_lag, max_lag)

    LOGGER.info('Computing beat tracking metrics.')
    beat_metrics_train = compute_beat_metrics(beat_times_train, beat_times_pred_train)
    beat_metrics_valid = compute_beat_metrics(beat_times_valid, beat_times_pred_valid)
    beat_metrics_test = compute_beat_metrics(beat_times_test, beat_times_pred_test)

    beat_metrics = {
        'train': beat_metrics_train,
        'valid': beat_metrics_valid,
        'test': beat_metrics_test,
    }

    beat_metrics_path = os.path.join(model_dir, 'beat_metrics.pkl')
    LOGGER.info('Saving beat tracking metrics.')
    with open(beat_metrics_path, 'wb') as f:
        pk.dump(beat_metrics, f)

    # Using test data, estimate tempo and evaluate
    LOGGER.info('Estimating tempo.')
    tempos_train = get_tempos_from_annotations(r, train_data['indices'])
    tempos_valid = get_tempos_from_annotations(r, valid_data['indices'])
    tempos_test = get_tempos_from_annotations(r, test_data['indices'])

    tempos_pred_train = estimate_tempos_for_batch(y_train_pred, frame_rate,
                                 min_lag, max_lag,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14)
    tempos_pred_valid = estimate_tempos_for_batch(y_valid_pred, frame_rate,
                                 min_lag, max_lag,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14)
    tempos_pred_test = estimate_tempos_for_batch(y_test_pred, frame_rate,
                                 min_lag, max_lag,
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
    tempo_metrics_path = os.path.join(model_dir, 'tempo_metrics.pkl')
    with open(tempo_metrics_path, 'wb') as f:
        pk.dump(tempo_metrics, f)

    LOGGER.info('Done!')


if __name__ == '__main__':
    main(**parse_arguments())
