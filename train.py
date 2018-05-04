import os
import numpy as np
import pickle as pk
from argparse import ArgumentParser
from hainsworth import prep_hainsworth_data
from preprocess_data import preprocess_data
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

HOP_SIZE =  0.01

HAINSWORTH_MIN_TEMPO = 40
HAINSWORTH_MAX_TEMPO = 250

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
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs for training model')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Batch size for training')
    parser.add_argument('--target-fs', type=int, default=44100,
                        help='Target sample rate. If spectrogram samples is used, must be 44100')
    parser.add_argument('--audio-window-size', type=int, default=2048,
                        help='Audio window size')
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
def main(data_dir, label_dir, dataset, output_dir, num_epochs=10, batch_size=5,
         lr=0.001, target_fs=44100, audio_window_size=2048, model_type='spectrogram'):
    """
    Train a deep beat tracker model
    """
    # Set up logger
    init_console_logger(LOGGER, verbose=True)

    output_dir = os.path.join(output_dir, model_type, dataset)
    LOGGER.info('Output will be saved to {}'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    LOGGER.info('Loading {} data.'.format(dataset))
    train_data_path = os.path.join(output_dir, '{}_train_data.npz').format(dataset)
    valid_data_path = os.path.join(output_dir, '{}_valid_data.npz').format(dataset)
    test_data_path = os.path.join(output_dir, '{}_test_data.npz').format(dataset)

    data_exists = os.path.exists(train_data_path) \
        and os.path.exists(valid_data_path) \
        and os.path.exists(test_data_path)

    if model_type == 'spectrogram':
        assert target_fs == 44100

    hop_length = int(target_fs * HOP_SIZE)

    # Load audio and annotations
    if dataset == 'hainsworth':
        a, r = prep_hainsworth_data(data_dir, label_dir, target_fs,
                                    load_audio=not data_exists)
    elif dataset == 'ballroom':
        a, r = prep_ballroom_data(data_dir, label_dir, hop_length, target_fs,
                                  load_audio=not data_exists)

    if not data_exists:
        # Create preprocessed data if it doesn't exist
        LOGGER.info('Preprocessing data for model type "{}".'.format(model_type))
        # Get features and targets from data
        X, y = preprocess_data(a, r, mode=model_type, hop_size=hop_length,
                               audio_window_size=audio_window_size, sr=target_fs)

        LOGGER.info('Creating data subsets.')
        train_data, valid_data, test_data = create_data_subsets(X, y)

        LOGGER.info('Saving data subsets to disk.')
        np.savez(train_data_path, **train_data)
        np.savez(valid_data_path, **valid_data)
        np.savez(test_data_path, **test_data)

    else:
        # Otherwise, just load existing data
        train_data = np.load(train_data_path)
        valid_data = np.load(valid_data_path)
        test_data = np.load(test_data_path)

    model_path = os.path.join(output_dir, 'model.hdf5')
    if not os.path.exists(model_path):
        # Only train model if we haven't done so already
        LOGGER.info('Training model.')
        # Create, train, and save model
        model_path = train_model(train_data, valid_data, model_type, model_path,
                                 lr=lr, batch_size=batch_size, num_epochs=num_epochs,
                                 audio_window_size=audio_window_size)

    LOGGER.info('Loading model.')
    model = load_model(model_path)

    frame_rate = target_fs / hop_length

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
    tempo_metrics_path = os.path.join(output_dir, 'tempo_metrics.pkl')
    with open(tempo_metrics_path, 'wb') as f:
        pk.dump(tempo_metrics, f)

    LOGGER.info('Done!')


if __name__ == '__main__':
    main(**parse_arguments())
