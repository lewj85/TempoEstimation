import os
import pickle as pk
from argparse import ArgumentParser
from hainsworth import prep_hainsworth_data
from preprocess_data import preprocess_data
from model.architectures import construct_spectrogram_bilstm
from model.training import train_model
from evaluation import compute_beat_metrics, compute_tempo_metrics
from data_utils import create_data_subsets
from beat_tracking import binary_targets_to_beat_times, \
    estimate_beats_from_activation
from tempo import get_tempos_from_annotations, estimate_tempo
from math import ceil

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


    # Creates a namespace object
    args = parser.parse_args()
    # vars creates a dictionary where key:value pairs attribute name:attribute
    return vars(args)


def main(data_dir, label_dir, dataset, output_dir, model_type='spectrogram'):
    """
    Train a deep beat tracker model
    """
    # Load audio and annotations
    if dataset == 'hainsworth':
        a, r = prep_hainsworth_data(data_dir, label_dir, TARGET_FS)
    elif dataset == 'ballroom':
        a, r = prep_ballroom_data(data_dir, label_dir, HOP_SIZE, TARGET_FS)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get features and targets from data
    X, y = preprocess_data(a, r, mode=model_type)

    train_data, valid_data, test_data = create_data_subsets(X, y)

    # Create, train, and save model
    model = train_model(train_data, valid_data, model_type,
                        lr=0.0001, batch_size=5, num_epochs=10)
    model_path = os.path.join(output_dir, 'model.h5')
    model.save(model_path)

    frame_rate = TARGET_FS / HOP_SIZE

    y_test_pred = model.predict(test_data['X'])

    # Using test data, estimate beats and evaluate
    beat_times_test = binary_targets_to_beat_times(test_data['y'], frame_rate)
    beat_times_pred = estimate_beats_for_batch(y_test_pred, frame_rate, HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG)
    beat_metrics = compute_beat_metrics(beat_times_test, beat_times_pred)
    beat_metrics_path = os.path.join(output_dir, 'beat_metrics.pkl')
    with open(beat_metrics_path, 'wb') as f:
        pk.dump(beat_metrics, f)

    # Using test data, estimate tempo and evaluate
    tempos_test = get_tempos_from_annotations(r, test_data['indices'])
    tempos_pred = estimate_tempos_for_batch(beat_act, frame_rate, 
                                 HAINSWORTH_MIN_LAG, HAINSWORTH_MAX_LAG,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14)
    tempo_metrics = compute_tempo_metrics(tempos_test, tempos_pred)
    tempo_metrics_path = os.path.join(output_dir, 'tempo_metrics.pkl')
    with open(tempo_metrics_path, 'wb') as f:
        pk.dump(tempo_metrics, f)


if __name__ == '__main__':
    main(**parse_arguments())
