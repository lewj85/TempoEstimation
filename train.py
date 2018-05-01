import os
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
    # Load audio and annotations
    if dataset == 'hainsworth':
        a, r = prep_hainsworth_data(data_dir, label_dir, TARGET_FS)
    elif dataset == 'ballroom':
        a, r = prep_ballroom_data(data_dir, label_dir, HOP_SIZE, TARGET_FS)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    LOGGER.info('Preprocessing data for model type "{}".'.format(model_type))
    # Get features and targets from data
    X, y = preprocess_data(a, r, mode=model_type)

    LOGGER.info('Creating data subsets.')
    train_data, valid_data, test_data = create_data_subsets(X, y)

    model_path = os.path.join(output_dir, 'model.hdf5')
    if not os.path.exists(model_path):
        LOGGER.info('Training model.')
        # Create, train, and save model
        model = train_model(train_data, valid_data, model_type, output_dir,
                            lr=0.0001, batch_size=5, num_epochs=10)
    else:
        LOGGER.info('Loading model.')
        model = load_model(model_path)

    frame_rate = TARGET_FS / HOP_SIZE

    LOGGER.info('Running model on test data.')
    y_test_pred = model.predict(test_data['X'])[:,:,1]

    # Using test data, estimate beats and evaluate
    LOGGER.info('Estimating beats.')
    beat_times_test = get_beat_times_from_annotations(r, test_data['indices'])
    beat_times_pred = estimate_beats_for_batch(y_test_pred, frame_rate, HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG)
    LOGGER.info('Computing beat tracking metrics.')
    beat_metrics = compute_beat_metrics(beat_times_test, beat_times_pred)
    beat_metrics_path = os.path.join(output_dir, 'beat_metrics.pkl')
    with open(beat_metrics_path, 'wb') as f:
        pk.dump(beat_metrics, f)

    # Using test data, estimate tempo and evaluate
    LOGGER.info('Estimating tempo.')
    tempos_test = get_tempos_from_annotations(r, test_data['indices'])
    tempos_pred = estimate_tempos_for_batch(y_test_pred, frame_rate,
                                 HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14)
    LOGGER.info('Computing tempo estimation metrics.')
    tempo_metrics = compute_tempo_metrics(tempos_test, tempos_pred)
    tempo_metrics_path = os.path.join(output_dir, 'tempo_metrics.pkl')
    with open(tempo_metrics_path, 'wb') as f:
        pk.dump(tempo_metrics, f)

    LOGGER.info('Done!')


if __name__ == '__main__':
    main(**parse_arguments())
