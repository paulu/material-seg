import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom

import config
from densecrf import densecrf_map
from descstore import DescriptorStoreMemmap
from scipy.misc import imsave


def labels_to_color(labels):
    assert labels_valid(labels)
    image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    image[:, :, :] = config.LABEL_COLORS[labels]
    return image


def labels_valid(labels):
    return (labels <= config.UNKNOWN_LABEL).all() and (labels >= 0).all()


def test_densecrf(image_dir, desc_dir, filepath_to_id_path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filepath_to_id = json.load(open(filepath_to_id_path))
    params = {
        "bilateral_pairwise_weight": 8,
        "bilateral_theta_lab_ab": 3.0,
        "bilateral_theta_lab_l": 0.5,
        "bilateral_theta_xy": 0.5,
        "min_dim": 550,
        "n_crf_iters": 10,
        "splat_triangle_weight": 1,
        "unary_prob_padding": 1e-05
    }

    desc_store = DescriptorStoreMemmap(desc_dir, readonly=True)
    # VGG-16
    stride = config.NETWORK_CONFIGS['209']['effective_stride']

    # Go through all images
    for filepath, img_id in filepath_to_id.iteritems():
        filename = os.path.basename(filepath)
        bname, ext = os.path.splitext(filename)
        img_path = os.path.join(image_dir, filename)
        image = np.array(Image.open(img_path))

        # Compute the expected output size
        h, w = image.shape[:2]
        prob_width = w // stride
        prob_height = h // stride

        img_id = int(img_id)
        prob = desc_store.get(img_id)
        prob = np.reshape(prob, (config.NLABELS, prob_height, prob_width))
        print prob.shape
        zoom_factor = (
            1,
            float(h) / prob_height,
            float(w) / prob_width,
        )
        prob_resized = zoom(prob, zoom=zoom_factor, order=1)
        labels_crf = densecrf_map(image, prob_resized.copy(), params)

        for l in range(config.NLABELS):
            img_mask = prob_resized[l, :, :][:, :, np.newaxis]
            red_img = np.array([255, 0, 0])[np.newaxis, np.newaxis, :]
            new_img = red_img * img_mask + image * (1 - img_mask)
            imsave(
                os.path.join(out_dir, '%s-prob-%s-crf%s' % (bname, config.LABEL_TO_NAME[l], ext)),
                new_img
            )
        imsave(
            os.path.join(out_dir, '%s-labels-crf%s' % (bname, ext)),
            labels_to_color(labels_crf)
        )
        imsave(os.path.join(out_dir, filename), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('desc_dir')
    parser.add_argument('filepath_to_id_path')
    parser.add_argument('out_dir')
    args = parser.parse_args()

    test_densecrf(
        args.image_dir, args.desc_dir, args.filepath_to_id_path, args.out_dir
    )
