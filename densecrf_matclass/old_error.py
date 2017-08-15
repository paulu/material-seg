import glob
import json
import multiprocessing
import os
import sys
import time
import traceback

import numpy as np
from scipy.misc import imsave
from skimage import transform

from caffe_matclass import config
from caffe_matclass.classify_material import ClassifyMaterial
from caffe_matclass.io import labels_to_color, labels_valid, load_gt_labels
from caffe_matclass.util import NumpyAwareJSONEncoder


def evaluate_error(image_filename, params):
    gt_fname = '%s-gt.png' % os.path.splitext(image_filename)[0]

    # predict raw probabilites
    cm = ClassifyMaterial(**params)
    predicted_labels = cm.predict(image_filename)

    if os.path.exists(gt_fname):
        gt_labels = load_gt_labels(gt_fname)
    else:
        print "WARNING: cannot find ground truth: %s" % gt_fname
        gt_labels = np.empty_like(predicted_labels)
        gt_labels.fill(config.UNKNOWN_LABEL)

    # resize ground truth to match prediction size
    tmp_scale = float(config.UNKNOWN_LABEL + 1)
    gt_labels = (transform.resize(gt_labels.astype(np.float) / tmp_scale, predicted_labels.shape, order=0, mode='nearest') * tmp_scale).astype(np.int)
    assert labels_valid(gt_labels)

    # save visualized ground truth
    if params['save_images']:
        gt_colored_fname = '%s-gt-color.png' % os.path.splitext(image_filename)[0]
        if not os.path.exists(gt_colored_fname):
            imsave(gt_colored_fname, labels_to_color(gt_labels))

    mask_image = (gt_labels != config.UNKNOWN_LABEL).astype(np.float)
    weight_image = config.CLASS_WEIGHTS[gt_labels] * mask_image
    #weight_image = mask_image
    if params['save_images']:
        weight_image_fname = '%s-gt-weight.png' % os.path.splitext(image_filename)[0]
        print 'saving %s...' % weight_image_fname
        imsave(weight_image_fname, weight_image)

    # measure accuracy
    total_weight = np.sum(weight_image)
    if total_weight > 0:
        error_image = (gt_labels != predicted_labels).astype(np.float) * mask_image
        if params['save_images']:
            error_image_fname = '%s-error-unweighted.png' % os.path.splitext(image_filename)[0]
            print 'saving %s...' % error_image_fname
            imsave(error_image_fname, error_image)

        if params['save_images']:
            error_colored = np.ones(gt_labels.shape)
            error_colored[gt_labels == config.UNKNOWN_LABEL] = 0.0
            error_colored[gt_labels == predicted_labels] *= 0.5
            error_colored_fname = '%s-error-colored.png' % os.path.splitext(image_filename)[0]
            print 'saving %s...' % error_colored_fname
            imsave(error_colored_fname, error_colored)

        error_weighted_image = error_image * weight_image
        if params['save_images']:
            error_weighted_image_fname = '%s-error-weighted.png' % os.path.splitext(image_filename)[0]
            print 'saving %s...' % error_weighted_image_fname
            imsave(error_weighted_image_fname, error_weighted_image)

        error = float(np.sum(error_weighted_image)), float(total_weight)
    else:
        error = None

    #print '%s: error: %s' % (image_filename, error)
    return error


def evaluate_error_wrapper(args):
    try:
        return evaluate_error(*args)
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        traceback.print_exc()
    return None


_best_mean_error = float('inf')
_num_error_samples = 0


def error_function(x):
    global _best_mean_error, _num_error_samples

    start_time = time.time()

    #print "error_function..."
    params = ClassifyMaterial.x_to_params(x)
    print 'x:',  x
    print 'params:', params

    # save images only on the first run
    params['save_images'] = (_num_error_samples == 0)
    #params['save_images'] = False
    _num_error_samples += 1

    image_filenames = glob.glob('images/*/*/*.jpg')
    #image_filenames = glob.glob('images/holdout/*/*.jpg')
    #image_filenames = glob.glob('images/houzz/*/*.jpg')
    #image_filenames = glob.glob('images/other/*/*.jpg')
    #print 'image_filenames:', image_filenames

    # dispatch
    worker_args = [(f, params) for f in image_filenames]
    num_processes = min(config.NUM_PROCESSES, len(image_filenames))
    if num_processes > 1:
        #print 'Using %s processes' % num_processes
        errors = multiprocessing.Pool(num_processes).map(
            evaluate_error_wrapper, worker_args)
    else:
        errors = map(evaluate_error_wrapper, worker_args)

    errors = filter(None, errors)
    if len(errors) != len(image_filenames):
        print "ERROR: LOST AN IMAGE: %s != %s" % (len(errors), len(image_filenames))
        print ''
        return 1e100
    mean_error = np.sum(x[0] for x in errors) / np.sum(x[1] for x in errors)
    mean_error_std = None

    if mean_error < _best_mean_error:
        json.dump(params, open('optimal_params.json', 'w'), indent=2, cls=NumpyAwareJSONEncoder)
        _best_mean_error = mean_error

    #print 'x:', x
    #print 'params:', params
    print 'mean_error: %s +/- %s' % (mean_error, mean_error_std)
    print 'best_mean_error:', _best_mean_error
    print 'time: %s sec' % (time.time() - start_time)
    print ''

    return mean_error
