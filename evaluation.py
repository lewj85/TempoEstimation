import os
import pickle as pk
import logging
import numpy as np
from math import ceil
import mir_eval.beat
from beat_tracking import estimate_beats_for_batch, \
    get_beat_times_from_annotations
from tempo import get_tempos_from_annotations, estimate_tempos_for_batch
from get_prior import get_tempo_prior

LOGGER = logging.getLogger('tempo_estimation')
LOGGER.setLevel(logging.DEBUG)

HOP_SIZE = 0.01
HAINSWORTH_MIN_TEMPO = 40
HAINSWORTH_MAX_TEMPO = 250


def create_metric_dict(values):
    return {
        'mean': np.mean(values),
        'stdev': np.std(values),
        'median': np.median(values),
        'min': np.min(values),
        'max': np.max(values),
        'values': list(values)
    }


def compute_beat_metrics(beat_times_ref, beat_times_est):
    """
    Compute standard MIREX beat tracking evaluation metrics
    """
    f_measure = []
    cemgil = []
    goto = []
    p_score = []

    for ref, est in zip(beat_times_ref, beat_times_est):
        f_measure.append(mir_eval.beat.f_measure(ref, est))
        cemgil.append(mir_eval.beat.cemgil(ref, est))
        goto.append(mir_eval.beat.goto(ref, est))
        p_score.append(mir_eval.beat.p_score(ref, est))

    metrics = {
        'f_measure': create_metric_dict(f_measure),
        'cemgil': create_metric_dict(cemgil),
        'goto': create_metric_dict(goto),
        'p_score': create_metric_dict(p_score),
    }

    return metrics


def compute_tempo_metrics(tempos_ref, tempos_est, tol=0.05):
    """
    Compute standard tempo accuracy metrics
    """
    tempos_ref = np.array(tempos_ref)
    tempos_est = np.array(tempos_est)

    ref_tol = tempos_ref * tol

    # Compute detections w.r.t. true tempo as well as multiples and fractions
    base_detection = np.abs(tempos_ref - tempos_est) <= ref_tol
    half_detection = np.abs(tempos_ref/2 - tempos_est) <= ref_tol
    double_detection = np.abs(tempos_ref*2 - tempos_est) <= ref_tol
    third_detection = np.abs(tempos_ref/3 - tempos_est) <= ref_tol
    triple_detection = np.abs(tempos_ref*3 - tempos_est) <= ref_tol
    multiple_detection = np.logical_or(base_detection,
                            np.logical_or(half_detection,
                                np.logical_or(double_detection,
                                    np.logical_or(third_detection,
                                                  triple_detection))))

    # Compute diffferent accuracy measures
    acc1 = np.mean(base_detection)
    acc2 = np.mean(multiple_detection)

    tempo_metrics = {
        'accuracy1': acc1,
        'accuracy2': acc2
    }

    return tempo_metrics


def perform_tempo_evaluations(tempos_train, tempos_valid, tempos_test,
    y_train_pred, y_valid_pred, y_test_pred,
    frame_rate, min_lag, max_lag, tempo_prior):
    """
    Evaluate tempo with a given tempo prior (or lack thereof)
    """

    # Using test data, estimate tempo and evaluate
    LOGGER.info('Estimating tempo.')
    tempos_pred_train = estimate_tempos_for_batch(y_train_pred, frame_rate,
                                 min_lag, max_lag,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14, tempo_prior=tempo_prior)
    tempos_pred_valid = estimate_tempos_for_batch(y_valid_pred, frame_rate,
                                 min_lag, max_lag,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14, tempo_prior=tempo_prior)
    tempos_pred_test = estimate_tempos_for_batch(y_test_pred, frame_rate,
                                 min_lag, max_lag,
                                 num_tempo_steps=100, alpha=0.79,
                                 smooth_win_len=.14, tempo_prior=tempo_prior)

    LOGGER.info('Computing tempo estimation metrics.')
    tempo_metrics_train = compute_tempo_metrics(tempos_train, tempos_pred_train)
    tempo_metrics_valid = compute_tempo_metrics(tempos_valid, tempos_pred_valid)
    tempo_metrics_test = compute_tempo_metrics(tempos_test, tempos_pred_test)

    tempos = {
        'train': tempos_pred_train,
        'valid': tempos_pred_valid,
        'test': tempos_pred_test
    }

    tempo_metrics = {
        'train': tempo_metrics_train,
        'valid': tempo_metrics_valid,
        'test': tempo_metrics_test,
    }

    return tempos, tempo_metrics


def perform_evaluation(train_data, valid_data, test_data, model_dir, r_train, r_test,
                       target_fs, batch_size, k_smoothing=1):
    """
    Evaluate the beat tracking model on beat tracking and tempo estimation
    """
    hop_length = int(target_fs * HOP_SIZE)
    min_lag = int(60 * target_fs / (hop_length * HAINSWORTH_MAX_TEMPO))
    max_lag = ceil(60 * target_fs / (hop_length * HAINSWORTH_MIN_TEMPO))

    frame_rate = target_fs / hop_length

    output_path = os.path.join(model_dir, 'output.npz')
    if not os.path.exists(output_path):
        from keras.models import load_model
        LOGGER.info('Loading model.')
        model_path = os.path.join(model_dir, 'model.hdf5')
        model = load_model(model_path)

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
        LOGGER.info('Saving model outputs.')
        np.savez(output_path, **outputs)
    else:
        data = np.load(output_path)
        y_train_pred = data['train']
        y_valid_pred = data['valid']
        y_test_pred = data['test']
        del data

    # Using test data, estimate beats and evaluate
    LOGGER.info('Getting annotation beats.')
    beat_times_train = get_beat_times_from_annotations(r_train, train_data['indices'])
    beat_times_valid = get_beat_times_from_annotations(r_train, valid_data['indices'])
    beat_times_test = get_beat_times_from_annotations(r_test, test_data['indices'])

    # Using test data, estimate beats and evaluate
    LOGGER.info('Estimating beats.')
    beat_times_pred_train = estimate_beats_for_batch(y_train_pred, frame_rate,
        min_lag, max_lag)
    beat_times_pred_valid = estimate_beats_for_batch(y_valid_pred, frame_rate,
        min_lag, max_lag)
    beat_times_pred_test = estimate_beats_for_batch(y_test_pred, frame_rate,
        min_lag, max_lag)

    beat_times = {
        'train': beat_times_pred_train,
        'valid': beat_times_pred_valid,
        'test': beat_times_pred_test
    }

    beat_times_path = os.path.join(model_dir, 'beat_times.pkl')
    LOGGER.info('Saving predicted beat times.')
    with open(beat_times_path, 'wb') as f:
        pk.dump(beat_times, f)

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
    LOGGER.info('Getting annotation tempo.')
    tempos_train = get_tempos_from_annotations(r_train, train_data['indices'])
    tempos_valid = get_tempos_from_annotations(r_train, valid_data['indices'])
    tempos_test = get_tempos_from_annotations(r_test, test_data['indices'])

    tempo_configs = [
        {},
        {'k': 0, 'apply_gaussian': False},
        {'k': k_smoothing, 'apply_gaussian': False},
        {'k': 0, 'apply_gaussian': True},
        {'k': k_smoothing, 'apply_gaussian': True},
    ]

    tempo_metrics = {}
    predicted_tempos = {}

    for conf in tempo_configs:
        if conf:
            desc = 'tempo_prior_k={}_{}gaussian'
            desc = desc.format(conf['k'],
                               "" if conf['apply_gaussian'] else "no")
            tempo_prior = get_tempo_prior([r_train[idx] for idx in train_data['indices']],
                target_sr=target_fs, hop_size=hop_length, min_lag=min_lag,
                max_lag=max_lag, **conf)
        else:
            desc = 'base'
            tempo_prior = None


        tempos, metrics = perform_tempo_evaluations(
            tempos_train, tempos_valid, tempos_test, y_train_pred, y_valid_pred,
            y_test_pred, frame_rate, min_lag, max_lag, tempo_prior)

        tempo_metrics[desc] = metrics
        predicted_tempos[desc] = tempos

    LOGGER.info('Saving predicted tempo.')
    tempos_path = os.path.join(model_dir, 'tempos.pkl')
    with open(tempos_path, 'wb') as f:
        pk.dump(predicted_tempos, f)

    LOGGER.info('Saving tempo estimation metrics.')
    tempo_metrics_path = os.path.join(model_dir, 'tempo_metrics.pkl')
    with open(tempo_metrics_path, 'wb') as f:
        pk.dump(tempo_metrics, f)
