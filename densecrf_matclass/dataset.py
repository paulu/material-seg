import glob
import os

import numpy as np
from scipy.misc import imread
from skimage import transform

import caffe
import config


def load_gt_labels(gt_fname):
    #gt_labels = config.GROUND_SET_TO_LABEL[imread(gt_fname)]
    gt_labels = imread(gt_fname)
    gt_labels[gt_labels == config.OTHER_LABEL] = config.UNKNOWN_LABEL
    assert labels_valid(gt_labels)
    return gt_labels


def load_gt_labels_for_image_fname(image_fname):
    gt_fname = os.path.join(
        'labels', '%09d.png' % (
            int(os.path.splitext(os.path.basename(image_fname))[0])
        )
    )
    return load_gt_labels(gt_fname)


def labels_to_color(labels):
    assert labels_valid(labels)
    image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    image[:, :, :] = config.LABEL_COLORS[labels]
    return image


def labels_valid(labels):
    return (labels <= config.UNKNOWN_LABEL).all() and (labels >= 0).all()


def resize_gt(gt_labels, new_shape):
    if gt_labels.shape != new_shape:
        # resize ground truth to match prediction size
        tmp_scale = float(config.UNKNOWN_LABEL + 1)
        gt_labels = (transform.resize(gt_labels.astype(np.float) / tmp_scale, new_shape, order=0, mode='nearest') * tmp_scale).astype(np.int)
    return gt_labels


def compute_input_shape(shape, params):
    max_dim = params.get('max_dim')
    if max_dim:
        if shape[0] > shape[1]:
            return (max_dim, int(max_dim * shape[1] / shape[0]))
        else:
            return (int(max_dim * shape[0] / shape[1]), max_dim)
    else:
        min_dim = params['min_dim']
        if shape[1] > shape[0]:
            return (min_dim, int(min_dim * shape[1] / shape[0]))
        else:
            return (int(min_dim * shape[0] / shape[1]), min_dim)


def load_dataset(params):
    dataset = []
    for image_fname in glob.glob('images/*.jpg'):
        image = caffe.io.load_image(image_fname)

        gt_labels = load_gt_labels_for_image_fname(image_fname)
        #print gt_labels.shape, gt_shape
        #assert gt_labels.shape == gt_shape
        #gt_labels[gt_labels == config.OTHER_LABEL] = config.UNKNOWN_LABEL
        assert labels_valid(gt_labels)

        #for l in xrange(config.NLABELS):
            #print "gt_fname: %s (%s): %s" % (
                #l, config.LABEL_TO_NAME[l],
                #(gt_labels == l).sum()
            #)

        dataset.append((image_fname, image, gt_labels))

    return dataset
