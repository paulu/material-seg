#!/usr/bin/env python2.7

import os

import numpy as np
from scipy.misc import imsave
from skimage import transform
from skimage.color import rgb2lab

import caffe
from caffe_matclass import config
from caffe_matclass.krahenbuhl2013.krahenbuhl2013 import DenseCRF
from caffe_matclass.io import labels_to_color
from caffe_matclass.util import roundint


class ClassifyMaterial(object):
    """
    Full prediction pipeline to label an image.
    Currently assumes AlexNet dimensions.
    """

    # these are the parameters to be optimized over when improving the parameters
    OPTIMIZE_PARAMS = [
        ('bilateral_theta_lab_l', 5.0, 'exp'),
        ('bilateral_theta_lab_ab', 5.0, 'exp'),
        ('bilateral_theta_xy', 0.05, 'exp'),
        ('bilateral_pairwise_weight', 100.0, 'exp'),
        ('position_pairwise_weight', 10, 'exp'),
        ('position_theta_xy', 0.05, 'exp'),
        ('scale_weight_pow', 0.0, 'linear'),
    ]

    # mapping from 'x' (solver vector) to parameter value.
    # v is the un-optimized value, x is the entry in the solution vector.
    OPTIMIZE_PARAM_MAPPINGS = {
        'exp': lambda v, x: v * np.exp(x),
        'linear': lambda v, x: v + x,
    }

    def __init__(self, **params):
        self.net = None
        self.save_images = True

        self.scales = (5.2, 5.1, 5.0, 4.9, 4.8)
        self.stride = 32
        self.receptive_field = 227
        self.input_pad = self.receptive_field // 2
        self.max_dim = self.receptive_field + self.stride*16

        self.unary_prob_padding = 1e-6
        self.scale_weight_pow = 0.0

        self.n_crf_iters = 10
        self.bilateral_theta_lab_l = 10.0
        self.bilateral_theta_lab_ab = 5.0
        self.bilateral_theta_xy = 0.10
        self.bilateral_pairwise_weight = 1000.0
        self.position_pairwise_weight = 10.0
        self.position_theta_xy = 0.05

        for k, v in params.iteritems():
            assert hasattr(self, k)
            setattr(self, k, v)

    @classmethod
    def x_to_params(cls, x):
        """ Convert an encoding "x" into the parameters """
        idx = 0
        params = {}
        for k, v, opname in cls.OPTIMIZE_PARAMS:
            op = cls.OPTIMIZE_PARAM_MAPPINGS[opname]
            if isinstance(v, np.ndarray):
                params[k] = op(v, x[idx:idx+len(v)])
                idx += len(v)
            else:
                params[k] = op(v, x[idx])
                idx += 1
        return params

    def predict(self, image_filename):
        """ Full prediction pipeline for one image.  Returns final labels. """

        #print 'predict...'
        self.image_filename = image_filename
        self.output_basename, _ = os.path.splitext(image_filename)
        #print 'load %s...' % image_filename
        self.im = caffe.io.load_image(image_filename)
        #print 'im.shape:', self.im.shape

        if self.im.shape[0] > self.im.shape[1]:
            self.prediction_shape = (self.max_dim, roundint(self.max_dim * self.im.shape[1] / self.im.shape[0]))
        else:
            self.prediction_shape = (roundint(self.max_dim * self.im.shape[0] / self.im.shape[1]), self.max_dim)

        im_resized_fname = '%s-input.png' % self.output_basename
        if not os.path.exists(im_resized_fname):
            imsave(im_resized_fname, transform.resize(self.im, self.prediction_shape))

        self.net_input_shapes = [
            (int(self.prediction_shape[0] * scale), int(self.prediction_shape[1] * scale))
            for scale in self.scales
        ]

        #fname = '%s-scale-probs.npz' % self.output_basename
        #if os.path.exists(fname):
            ##print 'loading %s...' % fname
            #f = np.load(fname)
            #scale_probs = [f[str(i)] for i, s in enumerate(self.scales)]
        #else:
            #scale_probs = self.predict_raw()
            #print 'saving %s...' % fname
            #np.savez(fname, **{str(i): p for i, p in enumerate(scale_probs)})
        scale_probs = self.predict_raw()

        crf_labels = self.densecrf(scale_probs)
        return crf_labels

    def load_net(self):
        print "load_net..."
        self.net = caffe.Net(
            os.path.join(config.MODEL_DIR, 'deploy_conv.prototxt'),
            os.path.join(config.MODEL_DIR, 'conv.caffemodel')
        )
        self.net.set_phase_test()
        self.net.set_channel_swap('data', (2, 1, 0))
        self.net.set_raw_scale('data', 255.0)
        mean_image = np.load(os.path.join(config.MODEL_DIR, 'ilsvrc_2012_mean.npy'))
        self.net.set_mean('data', mean_image, mode='channel')

        #assert self.net.blobs['data'].height == self.receptive_field
        #assert self.net.blobs['data'].width == self.receptive_field

    def predict_raw(self):
        """ Predict raw probabilities """

        print "predict_raw..."
        if not self.net:
            self.load_net()

        scale_probs = []
        for i, scale in enumerate(self.scales):
            height, width = self.net_input_shapes[i]
            batch_size = 2 if config.MIRROR_AVG else 1
            channels = 3

            print 'scale %s: %s, reshaping to (%s, %s, %s, %s)...' % (i, scale, batch_size, channels, height, width)
            self.net.blobs['data'].reshape(batch_size, channels, height, width)
            self.net.reshape()
            assert self.net.blobs['data'].num == batch_size
            assert self.net.blobs['data'].channels == channels
            assert self.net.blobs['data'].height == height
            assert self.net.blobs['data'].width == width

            print 'preprocessing (resampling input and subtracting mean)...'
            preprocessed = self.net.preprocess('data', self.im)
            print 'preprocessed.shape:', preprocessed.shape
            if config.MIRROR_AVG:
                print 'mirroring...'
                data = np.concatenate([
                    preprocessed[np.newaxis, :, :, :],
                    preprocessed[np.newaxis, :, :, ::-1],
                ], axis=0)
            else:
                data = preprocessed[np.newaxis, :, :, :]
            print 'data.shape:', data.shape

            print 'forward pass...'
            out = self.net.forward_all(data=data)
            if config.MIRROR_AVG:
                prob = 0.5 * (out['prob'][0, :, :, :] + out['prob'][1, :, :, ::-1])
            else:
                prob = out['prob'][0]
            print 'scale %s: output prob.shape: %s' % (i, prob.shape)

            #fc8_softmax = self.net.blobs['fc8-softmax'].data[0]
            #for l in xrange(fc8_softmax.shape[0]):
                #fname = self.output_basename + '-fc8-probs%s.png' % l
                #print 'saving %s...' % fname
                #imsave(fname, fc8_softmax[l, :, :])

            #data = self.net.blobs['data'].data[0]
            #self.net.deprocess('data', data)
            #fname = self.output_basename + '-data.png'
            #print 'saving %s...' % fname
            #imsave(fname, data)

            fname = '%s-scale%s-raw-labels.png' % (self.output_basename, i)
            if self.save_images:
                print 'saving %s...' % fname
                raw_labels_image = labels_to_color(prob.argmax(axis=0))
                raw_labels_image = raw_labels_image.astype(np.uint8)
                imsave(fname, raw_labels_image)
            scale_probs.append(prob)

        return scale_probs

    def densecrf(self, scale_probs):
        """ Take raw probabilities and smooth them with krahenbuhl2013 """
        #print "densecrf..."

        npixels = self.prediction_shape[0] * self.prediction_shape[1]
        nlabels = scale_probs[0].shape[0]
        #print 'nlabels:', nlabels
        #print 'npixels:', npixels

        #print 'prepare unary costs: -log probability...'
        unary_probs = np.empty((npixels, nlabels), dtype=np.float32)
        unary_probs.fill(1e-8)
        for i, prob in enumerate(scale_probs):
            #pad_x = ((self.receptive_field-1)/2) * self.prediction_shape[0] // self.net_input_shapes[i][0]
            #pad_y = ((self.receptive_field-1)/2) * self.prediction_shape[1] // self.net_input_shapes[i][1]
            #unpadded_prediction_shape = (
                #self.prediction_shape[0] - pad_x * 2,
                #self.prediction_shape[1] - pad_y * 2,
            #)
            for l in xrange(nlabels):
                unary_probs[:, l] += (
                    (self.scales[i] ** self.scale_weight_pow) *
                    transform.resize(prob[l, :, :], self.prediction_shape)
                ).ravel()
        unary_probs /= np.clip(np.sum(unary_probs, axis=1)[:, np.newaxis], 1e-6, 1e6)

        if self.save_images:
            for l in xrange(nlabels):
                fname = self.output_basename + '-probs%s.png' % l
                print 'saving %s...' % fname
                imsave(fname, unary_probs[:, l].reshape(self.prediction_shape[0:2]))

            fname = self.output_basename + '-maxprobs.png'
            print 'saving %s...' % fname
            maxprob_image = unary_probs.max(axis=1).reshape(self.prediction_shape[0:2])
            imsave(fname, maxprob_image)

            maxprob_labels_image = labels_to_color(unary_probs.argmax(axis=1).reshape(self.prediction_shape[0:2]))
            fname = self.output_basename + '-maxprobs-labels.png'
            imsave(fname, maxprob_labels_image)

            #fname = self.output_basename + '-maxprobs-weighted.png'
            #imsave(fname, maxprob_labels_image * maxprob_image[:, :, np.newaxis])

        unary_probs[:, config.OTHER_LABEL] = 0.0
        #unary_probs[:, config.SKIN_LABEL] = 0.0
        #unary_probs[:, config.HAIR_LABEL] = 0.0
        #unary_probs[:, config.PAPER_LABEL] = 0.0
        unary_costs = -np.log(unary_probs + self.unary_prob_padding)

        #print 'prepare features: L*, a*, b*, x, y...'

        if self.im.shape[0:2] != self.prediction_shape:
            im_lab = rgb2lab(transform.resize(self.im, self.prediction_shape[0:2]))
        else:
            im_lab = rgb2lab(self.im)

        maxdim = float(np.max(self.prediction_shape[0:2]))
        scaled_positions = np.indices(self.prediction_shape[0:2]).astype(np.float32)

        bilateral_features = np.zeros((npixels, 5), dtype=np.float32)
        bilateral_features[:, 0] = scaled_positions[0].ravel() / (self.bilateral_theta_xy * maxdim)
        bilateral_features[:, 1] = scaled_positions[1].ravel() / (self.bilateral_theta_xy * maxdim)
        bilateral_features[:, 2] = im_lab[:, :, 0].ravel() / self.bilateral_theta_lab_l
        bilateral_features[:, 3] = im_lab[:, :, 1].ravel() / self.bilateral_theta_lab_ab
        bilateral_features[:, 4] = im_lab[:, :, 2].ravel() / self.bilateral_theta_lab_ab

        position_features = np.zeros((npixels, 2), dtype=np.float32)
        bilateral_features[:, 0] = scaled_positions[0].ravel() / (self.position_theta_xy * maxdim)
        bilateral_features[:, 1] = scaled_positions[1].ravel() / (self.position_theta_xy * maxdim)

        #bilateral_pairwise_costs = (self.bilateral_pairwise_weight * (1 - np.identity(nlabels))).astype(np.float32)
        #position_pairwise_costs = (self.position_pairwise_weight * (1 - np.identity(nlabels))).astype(np.float32)

        #print 'dence crf inference (%s pixels, %s labels)...' % (npixels, nlabels)
        crf = DenseCRF(npixels, nlabels)
        crf.set_unary_energy(unary_costs)
        if self.bilateral_pairwise_weight > 0:
            crf.add_potts_pairwise_energy(self.bilateral_pairwise_weight, bilateral_features)
        if self.position_pairwise_weight > 0:
            crf.add_potts_pairwise_energy(self.position_pairwise_weight, position_features)
        map_result = crf.map(n_iters=self.n_crf_iters)
        crf_labels = map_result.reshape((self.prediction_shape[0:2]))

        if self.save_images:
            print 'crf_labels.shape:', crf_labels.shape
            crf_labels_image = labels_to_color(crf_labels)
            fname = '%s-crf-labels.png' % self.output_basename
            print 'saving %s...' % fname
            imsave(fname, crf_labels_image)

        return crf_labels
