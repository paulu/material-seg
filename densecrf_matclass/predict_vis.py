import multiprocessing
import os
import shutil
import traceback

import numpy as np
from scipy.misc import imsave

import caffe
from caffe_matclass import config
from caffe_matclass.dataset import compute_input_shape, labels_to_color
from caffe_matclass.densecrf import densecrf_map
from caffe_matclass.predict import (forward_prob_footprints, load_net,
                                    prob_footprints_to_unary)
from caffe_matclass.util import mkdir_p


def imsave_v(fname, image):
    print "Saving %s..." % fname
    imsave(fname, image)


def vis(image_fnames, net_ids, nprocesses=None, gpu=False, save_pred=True, return_pred=True):
    args = [(image_fname, net_ids, gpu, save_pred, return_pred) for image_fname in image_fnames]
    if nprocesses == 1:
        ret = map(vis_worker, args)
    else:
        # hack to fix processor affinity -- see http://stackoverflow.com/questions/15639779/what-determines-whether-different-python-processes-are-assigned-to-the-same-or-d
        os.system("taskset -p 0xffffffffffffffff %d" % os.getpid())
        pool = multiprocessing.Pool(nprocesses)
        ret = pool.map(vis_worker, args)
    return (ret if return_pred else None)


def vis_worker(args):
    print "vis_worker:", args
    try:
        return vis_worker_impl(args)
    except:
        traceback.print_exc()
        raise


def vis_worker_impl((image_filename, net_ids, gpu, save_pred, return_pred)):
    outdir = os.path.join(
        'results', os.path.splitext(os.path.basename(image_filename))[0])

    pred_fname = os.path.join(outdir, 'pred.npy')
    if os.path.exists(pred_fname):
        pred = np.load(pred_fname)
    else:
        image = caffe.io.load_image(image_filename)

        labels_shape = compute_input_shape(image.shape, config.DENSECRF_CONFIG)
        pred = np.zeros((config.NLABELS, labels_shape[0], labels_shape[1]))

        for net_id in net_ids:
            params = config.NETWORK_CONFIGS[net_id]
            net = load_net(params, gpu=gpu)
            prob_footprints = forward_prob_footprints(image, net, params)
            netdir = os.path.join(outdir, 'net-%s' % net_id)
            pred += process_results(netdir, image, image_filename, prob_footprints, params)

        visualize_results(outdir, image, image_filename, pred, params)
        if save_pred:
            np.save(pred_fname, pred.astype(np.float16))

    return (pred if return_pred else None)


def visualize_results(outdir, image, image_filename, pred, params):
    mkdir_p(outdir)

    # normalize
    pred = pred / np.clip(np.sum(pred, axis=0), 1e-20, 1e20)[np.newaxis, :, :]

    labels_maxprob = labels_to_color(pred.argmax(axis=0))
    labels_maxprob[np.max(pred, axis=0) < 1e-10] = 0
    imsave_v(os.path.join(outdir, 'labels-maxprob.png'), labels_maxprob)
    for l in xrange(pred.shape[0]):
        imsave_v(os.path.join(outdir, 'prob-%s.png' % l), pred[l, :, :])

    for thresh in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        labels_t = np.copy(labels_maxprob)
        labels_t[np.max(pred, axis=0) < (thresh / 10.0)] = 0
        imsave_v(os.path.join(outdir, 'labels-%s.png' % thresh), labels_t)

    shutil.copyfile(image_filename, os.path.join(outdir, os.path.basename(image_filename)))

    for weight in (1, 2, 5, 8, 10, 100):
        densecrf_config = config.DENSECRF_CONFIG.copy()
        densecrf_config['bilateral_pairwise_weight'] *= weight
        labels_crf = densecrf_map(image, pred, densecrf_config)
        imsave_v(os.path.join(outdir, 'labels-crf-%s.png' % weight), labels_to_color(labels_crf))


def process_results(outdir, image, image_filename, prob_footprints, params):
    mkdir_p(outdir)

    with open(os.path.join(outdir, 'sample.txt'), 'w') as f:
        for i, p in enumerate(prob_footprints):
            # visualize splats
            y0 = int(np.clip(p['foot'][0], 0, 1) * image.shape[0])
            x0 = int(np.clip(p['foot'][1], 0, 1) * image.shape[1])
            y1 = int(np.clip(p['foot'][2], 0, 1) * image.shape[0])
            x1 = int(np.clip(p['foot'][3], 0, 1) * image.shape[1])
            imsave_v(os.path.join(outdir, 'sample%03d-image.png' % i), image[y0:y1, x0:x1, :])
            imsave_v(os.path.join(outdir, 'sample%03d-maxprob.png' % i), labels_to_color(p['prob'].argmax(axis=0)))

            print >>f, 'sample %s: foot: %s' % (i, p['foot'])

    labels_shape = compute_input_shape(image.shape, config.DENSECRF_CONFIG)
    pred = prob_footprints_to_unary(prob_footprints, labels_shape, params)
    visualize_results(outdir, image, image_filename, pred, params)

    # average
    return pred / len(prob_footprints)

