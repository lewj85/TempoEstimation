import numpy as np
import mir_eval.beat


def create_metric_dict(values):
    return {
        'mean': np.mean(values),
        'stdev': np.std(values),
        'median': np.median(values),
        'min': np.min(values),
        'max': np.max(values),
        'values': values
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
        try:
            p_score.append(mir_eval.beat.p_score(ref, est))
        except:
            import pdb
            pdb.set_trace()

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
