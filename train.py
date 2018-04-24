from argparse import ArgumentParser
from hainsworth import prep_hainsworth_data
from preprocess_data import preprocess_data
from model.architectures import construct_spectrogram_bilstm
from model.training import train_model

HOP_SIZE = 512
TARGET_FS = 44100


def parse_arguments():
    """
    Get command line arguments
    """
    parser = ArgumentParser(description='Train a deep beat tracker model')
    parser.add_argument('data_dir', help='Path to audio directory')
    parser.add_argument('label_dir', help='Path to annotation directory')
    parser.add_argument('dataset', choices=['hainsworth', 'ballroom'],
                        help='Name of dataset')
    # -m is shortcut for --model
    parser.add_argument('--model', '-m', dest='model_type', default='spectrogram',
                        choices=['spectrogram', 'audio'], help='Model type')


    # Creates a namespace object
    args = parser.parse_args()
    # vars creates a dictionary where key:value pairs attribute name:attribute
    return vars(args)


def main(data_dir, label_dir, dataset, model_type='spectrogram'):
    """
    Train a deep beat tracker model
    """
    # Load audio and annotations
    if dataset == 'hainsworth':
        a, r = prep_hainsworth_data(data_dir, label_dir, TARGET_FS)
    elif dataset == 'ballroom':
        a, r = prep_ballroom_data(data_dir, label_dir, HOP_SIZE, TARGET_FS)

    # Get features and targets from data
    X, y = preprocess_data(a, r, mode=model_type)

    # Create and train model
    model = train_model(X, y, model_type)


if __name__ == '__main__':
    main(**parse_arguments())
