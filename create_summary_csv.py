import os
import pickle
import json
import csv
from argparse import ArgumentParser

SUBSET_NAMES = ['train', 'valid', 'test']
BEAT_METRIC_NAMES = ['f_measure', 'cemgil', 'goto', 'p_score']
STAT_NAMES = ['mean', 'stdev', 'median', 'min', 'max']
TEMPO_VARIANTS = ['base', 'tempo_prior_k=0_nogaussian', 'tempo_prior_k=1_nogaussian', 'tempo_prior_k=0_gaussian', 'tempo_prior_k=1_gaussian']


def parse_arguments():
    parser = ArgumentParser(description='Create an output summary file')
    parser.add_argument('output_data_dir', help='Path to output data directory')
    parser.add_argument('output_dir', help='Path to output summary file directory', default='./', nargs='?')
    args = parser.parse_args()
    return vars(args)


def create_summary_csv(output_data_dir, output_dir):
    fieldnames = ['dataset', 'num_epochs', 'batch_size', 'lr',
                  'patience', 'target_fs', 'audio_window_size',
                  'k_smoothing', 'model_type']
    for subset_name in SUBSET_NAMES:
        for metric_name in BEAT_METRIC_NAMES:
            for stat_name in STAT_NAMES:
                field_name = "{}_{}_{}".format(subset_name, metric_name, stat_name)
                fieldnames.append(field_name)
    for subset_name in SUBSET_NAMES:
        for variant_name in TEMPO_VARIANTS:
            fieldnames.append('{}_{}_accuracy1'.format(subset_name, variant_name))
            fieldnames.append('{}_{}_accuracy2'.format(subset_name, variant_name))

    with open(os.path.join(output_dir, 'summary.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_type_dir in next(os.walk(output_data_dir))[1]:
            for dataset_dir in next(os.walk(output_data_dir+'/'+model_type_dir+'/'))[1]:
                for file_dir in next(os.walk(os.path.join(output_data_dir,model_type_dir,dataset_dir,'model')))[1]:
                    load_and_write(os.path.join(output_data_dir,model_type_dir,dataset_dir,'model',file_dir), writer)


def load_and_write(file_dir, writer):
    config_path = os.path.join(file_dir, 'config.json')
    beat_metrics_path = os.path.join(file_dir, 'beat_metrics.pkl')
    tempo_metrics_path = os.path.join(file_dir, 'tempo_metrics.pkl')

    if not (os.path.exists(config_path) \
            and os.path.exists(beat_metrics_path) \
            and os.path.exists(tempo_metrics_path)):
        warn_msg = "Incomplete training output directory: {}. Skipping."
        print(warn_msg.format(file_dir))
        return

    with open(config_path, 'r') as f:
        c = json.load(f)
    with open(beat_metrics_path, 'rb') as f:
        b = pickle.load(f)
    with open(tempo_metrics_path, 'rb') as f:
        t = pickle.load(f)

    row = {}

    row['dataset'] = c['dataset']
    row['num_epochs'] = c['num_epochs']
    row['batch_size'] = c['batch_size']
    row['lr'] = c['lr']
    row['patience'] = c.get('patience', 5)
    row['target_fs'] = c['target_fs']
    row['audio_window_size'] = c['audio_window_size']
    row['k_smoothing'] = c.get('k_smoothing', 1)
    row['model_type'] = c['model_type']

    # Define keys ahead of time to ensure ordering

    for subset_name in SUBSET_NAMES:
        for metric_name in BEAT_METRIC_NAMES:
            for stat_name in STAT_NAMES:
                field_name = "{}_{}_{}".format(subset_name, metric_name, stat_name)
                row[field_name] = b[subset_name][metric_name][stat_name]

    for subset_name in SUBSET_NAMES:
        for variant_name in TEMPO_VARIANTS:
            row['{}_{}_accuracy1'.format(subset_name, variant_name)] \
                = t[variant_name][subset_name]['accuracy1']
            row['{}_{}_accuracy2'.format(subset_name, variant_name)] \
                = t[variant_name][subset_name]['accuracy2']

    writer.writerow(row)


if __name__ == '__main__':
    create_summary_csv(**parse_arguments())
