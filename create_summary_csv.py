import os
import pickle
import json
import csv
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser(description='Create an output summary file')
    parser.add_argument('output_data_dir', help='Path to output data directory')
    parser.add_argument('output_dir', help='Path to output summary file directory', default='./')
    args = parser.parse_args()
    return vars(args)


def create_summary_csv(output_data_dir, output_dir):
    with open(os.path.join(output_dir, 'summary.csv'), 'w') as f:
        fieldnames = ['dataset', 'num_epochs', 'batch_size', 'lr',
                      'patience', 'target_fs', 'audio_window_size',
                      'k_smoothing', 'model_type',
                      'mean', 'stdev', 'median', 'min', 'max',
                      'train', 'valid', 'test']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_type_dir in next(os.walk(output_data_dir))[1]:
            for dataset_dir in next(os.walk(output_data_dir+'/'+model_type_dir+'/'))[1]:
                for file_dir in next(os.walk(os.path.join(output_data_dir,model_type_dir,dataset_dir,'model')))[1]:
                    load_and_write(os.path.join(output_data_dir,model_type_dir,dataset_dir,'model',file_dir), writer)


def load_and_write(file_dir, writer):
    c = json.load(file_dir + '/config.json')
    b = pickle.load(file_dir + '/beat_metrics.pkl')
    t = pickle.load(file_dir + '/tempo_metrics.pkl')

    writer.writerow({'dataset': c['dataset'],
                     'num_epochs': c['num_epochs'],
                     'batch_size': c['batch_size'],
                     'lr': c['lr'],
                     'patience': c['patience'],
                     'target_fs': c['target_fs'],
                     'audio_window_size': c['audio_window_size'],
                     'k_smoothing': c['k_smoothing'],
                     'model_type': c['model_type'],
                     'train_f_measure_mean': b['train']['f_measure']['mean'],
                     'train_f_measure_stdev': b['train']['f_measure']['stdev'],
                     'train_f_measure_median': b['train']['f_measure']['median'],
                     'train_f_measure_min': b['train']['f_measure']['min'],
                     'train_f_measure_max': b['train']['f_measure']['max'],
                     'train_cemgil_mean': b['train']['cemgil']['mean'],
                     'train_cemgil_stdev': b['train']['cemgil']['stdev'],
                     'train_cemgil_median': b['train']['cemgil']['median'],
                     'train_cemgil_min': b['train']['cemgil']['min'],
                     'train_cemgil_max': b['train']['cemgil']['max'],
                     'train_goto_mean': b['train']['goto']['mean'],
                     'train_goto_stdev': b['train']['goto']['stdev'],
                     'train_goto_median': b['train']['goto']['median'],
                     'train_goto_min': b['train']['goto']['min'],
                     'train_goto_max': b['train']['goto']['max'],
                     'train_p_score_mean': b['train']['p_score']['mean'],
                     'train_p_score_stdev': b['train']['p_score']['stdev'],
                     'train_p_score_median': b['train']['p_score']['median'],
                     'train_p_score_min': b['train']['p_score']['min'],
                     'train_p_score_max': b['train']['p_score']['max'],
                     'valid_f_measure_mean': b['valid']['f_measure']['mean'],
                     'valid_f_measure_stdev': b['valid']['f_measure']['stdev'],
                     'valid_f_measure_median': b['valid']['f_measure']['median'],
                     'valid_f_measure_min': b['valid']['f_measure']['min'],
                     'valid_f_measure_max': b['valid']['f_measure']['max'],
                     'valid_cemgil_mean': b['valid']['cemgil']['mean'],
                     'valid_cemgil_stdev': b['valid']['cemgil']['stdev'],
                     'valid_cemgil_median': b['valid']['cemgil']['median'],
                     'valid_cemgil_min': b['valid']['cemgil']['min'],
                     'valid_cemgil_max': b['valid']['cemgil']['max'],
                     'valid_goto_mean': b['valid']['goto']['mean'],
                     'valid_goto_stdev': b['valid']['goto']['stdev'],
                     'valid_goto_median': b['valid']['goto']['median'],
                     'valid_goto_min': b['valid']['goto']['min'],
                     'valid_goto_max': b['valid']['goto']['max'],
                     'valid_p_score_mean': b['valid']['p_score']['mean'],
                     'valid_p_score_stdev': b['valid']['p_score']['stdev'],
                     'valid_p_score_median': b['valid']['p_score']['median'],
                     'valid_p_score_min': b['valid']['p_score']['min'],
                     'valid_p_score_max': b['valid']['p_score']['max'],
                     'test_f_measure_mean': b['test']['f_measure']['mean'],
                     'test_f_measure_stdev': b['test']['f_measure']['stdev'],
                     'test_f_measure_median': b['test']['f_measure']['median'],
                     'test_f_measure_min': b['test']['f_measure']['min'],
                     'test_f_measure_max': b['test']['f_measure']['max'],
                     'test_cemgil_mean': b['test']['cemgil']['mean'],
                     'test_cemgil_stdev': b['test']['cemgil']['stdev'],
                     'test_cemgil_median': b['test']['cemgil']['median'],
                     'test_cemgil_min': b['test']['cemgil']['min'],
                     'test_cemgil_max': b['test']['cemgil']['max'],
                     'test_goto_mean': b['test']['goto']['mean'],
                     'test_goto_stdev': b['test']['goto']['stdev'],
                     'test_goto_median': b['test']['goto']['median'],
                     'test_goto_min': b['test']['goto']['min'],
                     'test_goto_max': b['test']['goto']['max'],
                     'test_p_score_mean': b['test']['p_score']['mean'],
                     'test_p_score_stdev': b['test']['p_score']['stdev'],
                     'test_p_score_median': b['test']['p_score']['median'],
                     'test_p_score_min': b['test']['p_score']['min'],
                     'test_p_score_max': b['test']['p_score']['max'],
                     'train_accuracy1': t['train']['accuracy1'],
                     'train_accuracy2': t['train']['accuracy2'],
                     'valid_accuracy1': t['valid']['accuracy1'],
                     'valid_accuracy2': t['valid']['accuracy2'],
                     'test_accuracy1': t['test']['accuracy1'],
                     'test_accuracy2': t['test']['accuracy2']})


if __name__ == '__main__':
    create_summary_csv(**parse_arguments())
