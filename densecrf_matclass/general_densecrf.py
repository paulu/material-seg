import numpy as np
from skimage import transform
from skimage.color import rgb2lab

from krahenbuhl2013.krahenbuhl2013 import DenseCRF


def densecrf_map(image, probs, params):
    lcrf = LearnableDenseCRF(image, probs, params)
    return lcrf.map(params)


class LearnableDenseCRF(object):

    def __init__(self, image, pred, params):
        """ pred: [C x H x W] """

        #assert pred.shape[1:3] == prediction_shape
        prediction_shape = pred.shape[1:3]

        self.prediction_shape = prediction_shape
        self.npixels = np.prod(prediction_shape[0:2])
        self.nlabels = pred.shape[0]
        self.params = params

        # Convert from [C x H x W] to [(H*W) x C]
        unary_probs = pred.reshape(self.nlabels, self.npixels).transpose()
        # remove 'other' class
        #unary_probs[:, config.NAME_TO_LABEL['other']] = 0.0

        self.unary_costs = -np.log(np.clip(unary_probs + params['unary_prob_padding'], 1e-20, 1e20))
        self.unary_costs = np.copy(self.unary_costs, order='C').astype(np.float32)

        if image.shape[0:2] == prediction_shape:
            self.im_lab = rgb2lab(image)
        else:
            self.im_lab = rgb2lab(transform.resize(image, prediction_shape[0:2]))

        # scale features to have have dynamic range ~10-20ish
        self.scaled_positions = (
            np.indices(prediction_shape[0:2]).astype(np.float32) *
            10.0 / float(np.min(prediction_shape[0:2]))
        )
        self.bilateral_features = np.zeros((self.npixels, 5), dtype=np.float32)
        self.bilateral_features[:, 0] = self.scaled_positions[0].ravel()
        self.bilateral_features[:, 1] = self.scaled_positions[1].ravel()
        self.bilateral_features[:, 2] = self.im_lab[:, :, 0].ravel() / 10.0
        self.bilateral_features[:, 3] = self.im_lab[:, :, 1].ravel() / 10.0
        self.bilateral_features[:, 4] = self.im_lab[:, :, 2].ravel() / 10.0

    def set_gt(self, gt):
        assert gt.shape == self.prediction_shape
        self.gt = gt.astype(np.int32)

    def _build(self, params):
        kernel_params = np.array([
            params['bilateral_theta_xy'],
            params['bilateral_theta_xy'],
            params['bilateral_theta_lab_l'],
            params['bilateral_theta_lab_ab'],
            params['bilateral_theta_lab_ab'],
        ], dtype=np.float32)

        self.crf = DenseCRF(self.npixels, self.nlabels)
        self.crf.set_unary_energy(self.unary_costs)
        if params['bilateral_pairwise_weight'] > 0:
            self.crf.add_potts_pairwise_energy(
                params['bilateral_pairwise_weight'],
                self.bilateral_features,
                kernel_params,
            )

    def map(self, params):
        """ MAP inference """
        self._build(params)
        map_result = self.crf.map(n_iters=params['n_crf_iters'])
        crf_labels = map_result.reshape((self.prediction_shape[0:2]))
        return crf_labels


