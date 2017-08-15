import os

import numpy as np
from skimage import transform

from caffe.classifier import Classifier
from caffe_matclass import config
from caffe_matclass.dataset import compute_input_shape


def prepare_net_for_test(net):
    net.set_phase_test()
    net.set_channel_swap('data', (2, 1, 0))
    net.set_raw_scale('data', 255.0)
    mean_image = np.load(os.path.join(
        os.path.dirname(__file__),
        'data', 'ilsvrc_2012_mean.npy'))
    net.set_mean('data', mean_image, mode='channel')
    return net


def load_net(params, model_dir='.', gpu=False, device_id=0):
    def fname(k):
        return str(os.path.join(model_dir, params[k]))

    if 'deploy' in params:
        # network is in correct format
        net_target = Classifier(fname('deploy'), fname('caffemodel'))
    else:
        # we need to transplant weights from source to target
        net_source = Classifier(fname('deploy_source'), fname('caffemodel'))
        net_target = Classifier(fname('deploy_target'), fname('caffemodel'))

        for t, s in params['param_mapping'].iteritems():
            if not s:
                continue
            for blob_idx in (0, 1):
                print '%s %s %s <-- %s %s %s' % (
                    t, blob_idx, net_target.params[t][blob_idx].data.shape,
                    s, blob_idx, net_source.params[s][blob_idx].data.shape,
                )
                net_target.params[t][blob_idx].data[...] = (
                    np.reshape(
                        net_source.params[s][blob_idx].data,
                        net_target.params[t][blob_idx].data.shape
                    )
                )

        #with NamedTemporaryFile() as f:
        #    net_target.save(f.name)
        #    net_target = Classifier(fname('deploy_target'), f.name)

    if gpu:
        print "set_mode_gpu"
        net_target.set_mode_gpu()
        net_target.set_device(device_id)
    else:
        print "set_mode_cpu"
        net_target.set_mode_cpu()

    prepare_net_for_test(net_target)
    return net_target


def net_forward(net, data_dict, params):
    final_data_dict = {}

    for k, v in data_dict.iteritems():
        if params.get('mirror'):
            print 'mirroring...'
            final_data_dict[k] = np.concatenate([
                v[np.newaxis, :, :, :],
                v[np.newaxis, :, :, ::-1],
            ], axis=0)
        else:
            final_data_dict[k] = v[np.newaxis, :, :, :]

    print 'forward input shapes: ', [(k, v.shape) for (k, v) in final_data_dict.iteritems()]
    out = net.forward_all(**final_data_dict)
    if params.get('mirror'):
        prob = 0.5 * (out['prob'][0, :, :, :] + out['prob'][1, :, :, ::-1])
    else:
        prob = out['prob'][0]
    print 'forward output shape: %s, min: %s, max: %s' % (prob.shape, np.min(prob), np.max(prob))
    return prob


def forward_prob_footprints(image, net, params):
    if params['mode'] == 'sliding':
        return forward_prob_sliding_footprints(image, net, params)
    elif params['mode'] == 'grid':
        return forward_prob_grid_footprints(image, net, params)
    else:
        raise ValueError("Unknown mode: %s" % params['mode'])


def estimate_net_effective_stride(net, receptive_field):
    """ Estimate the net's effective stride by slowly increasing the input size
    until the output size increases """
    stride = 1
    while True:
        data_shape = (1, 3, receptive_field + stride, receptive_field + stride)
        net.blobs['data'].reshape(*data_shape)
        net.reshape()
        if net.blobs['prob'].width > 2 and net.blobs['prob'].height > 2:
            return stride - 1
        stride += 1


def forward_prob_sliding_footprints(image, net, params):
    # compute scale we want, ignoring whether or not it fits in an integer
    # number of locations
    base_height, base_width = compute_input_shape(image.shape, params)
    batch_size = 2 if params.get('mirror') else 1
    stride = params['effective_stride']

    # scale oversample
    nscales = params.get('oversample_scale', 3)
    if nscales == 1:
        scales = [1]
    else:
        scales = np.exp(np.linspace(np.log(1/np.sqrt(2)), np.log(np.sqrt(2)), nscales))

    # padding oversample
    npads = params.get('oversample_pad', 1)
    if npads == 1:
        extra_pads = [0]
    else:
        extra_pads = stride * np.arange(npads) / npads

    prob_footprints = []
    for scale in scales:
        scale_height = (base_height * scale)
        scale_width = (base_width * scale)

        # use a slightly smaller scale to give us an integer number of sliding
        # locations across the photo
        off = params['receptive_field'] - 2 * params['input_pad']
        height = int(((scale_height - off) // stride) * stride + off)
        width = int(((scale_width - off) // stride) * stride + off)

        if image.shape[0:2] == (height, width):
            image_resized = image
        else:
            image_resized = transform.resize(image, (height, width))

        # pad the end of the image to allow for sliding off the end with the
        # extra_pad offsets
        if len(extra_pads) > 1:
            image_resized = np.pad(
                image_resized,
                ((0, extra_pads[-1]), (0, extra_pads[-1]), (0, 0)),
                mode='mean',
            )

        num_chunks = params.get('num_chunks', 1)
        if num_chunks != 1:
            raise NotImplementedError("num_chunks != 1: not implemented")

        data_shape = (batch_size, 3, height, width)
        print 'reshaping to %s...' % (data_shape, )
        net.blobs['data'].reshape(*data_shape)
        net.reshape()

        for extra_pad_x in extra_pads:
            for extra_pad_y in extra_pads:
                y0 = extra_pad_y
                x0 = extra_pad_x
                y1 = height + extra_pad_y
                x1 = width + extra_pad_x

                #assert 0 <= x0 <= width and 0 <= y0 <= height
                #assert 0 <= x1 <= width and 0 <= y1 <= height
                #assert x0 < x1 and y0 < y1

                print 'preprocessing (resampling input and subtracting mean)...'
                data_dict = {
                    'data': net.preprocess('data', image_resized[y0:y1, x0:x1, :])
                }
                prob_footprints.append({
                    'foot': (float(y0) / height, float(x0) / width, float(y1) / height, float(x1) / width),
                    'prob': net_forward(net, data_dict, params)
                })

    return prob_footprints


def forward_prob_grid_footprints(image, net, params):
    """ Return footprints """
    base_height, base_width = compute_input_shape(image.shape, params)
    data_shape = net.blobs['data'].data.shape[2:4]

    # scale oversample
    nscales = params.get('oversample_scale', 1)
    if nscales == 1:
        scales = [1]
    else:
        min_scale = params.get('min_scale', 1.0/np.sqrt(2))
        max_scale = params.get('max_scale', np.sqrt(2))
        scales = np.exp(np.linspace(np.log(min_scale), np.log(max_scale), nscales))

    prob_footprints = []
    for scale in scales:
        height = int(base_height * scale)
        width = int(base_width * scale)

        if image.shape[0:2] == (height, width):
            image_resized = image
        else:
            image_resized = transform.resize(image, (height, width))

        # spatial oversample
        nx = params.get('oversample_x', 1)
        ny = params.get('oversample_y', 1)
        if nx == 1 and ny == 1:
            # center crop only
            coords = [
                ((height - data_shape[0]) / 2, (width - data_shape[1]) / 2)
            ]
        else:
            # grid of crops
            coords = []
            for fy in np.linspace(0, 1, ny):
                for fx in np.linspace(0, 1, nx):
                    coords.append((
                        int(fy * (height - data_shape[0])),
                        int(fx * (width - data_shape[1]))
                    ))

        for (y0, x0) in coords:
            y1 = y0 + data_shape[0]
            x1 = x0 + data_shape[1]
            image_cropped = image_resized[y0:y1, x0:x1, :]

            data_dict = {
                'data': net.preprocess('data', image_cropped)
            }

            if params.get('has_global_data'):
                data_dict['g-data'] = net.preprocess('g-data', image_cropped)

            foot = (float(y0) / height, float(x0) / width, float(y1) / height, float(x1) / width)
            print 'foot:', foot
            prob = net_forward(net, data_dict, params)
            prob_footprints.append({'foot': foot, 'prob': prob})

    return prob_footprints


def splat_prob(prob, foot, target, params):
    nlabels = prob.shape[0]

    splat_triangle_weight = params.get('splat_triangle_weight', 0)
    splat_nearest_weight = params.get('splat_nearest_weight', 0)
    splat_dirac_weight = params.get('splat_dirac_weight', 0)

    y0 = int(foot[0] * target.shape[1])
    x0 = int(foot[1] * target.shape[2])
    y1 = int(foot[2] * target.shape[1])
    x1 = int(foot[3] * target.shape[2])
    foot_shape = (y1 - y0, x1 - x0)
    #print "target.shape: %s, foot: %s, foot.shape %s" % (target.shape, foot, foot_shape)
    assert foot_shape[0] > 0 and foot_shape[1] > 0
    assert prob.shape[0] == target.shape[0]

    # optionally pad each prediction with 0s.  For example,
    # splat_interleave = 3 does this:
    # [1 2] --> [0 0 0 0 0 0]
    # [3 4]     [0 1 0 0 2 0]
    #           [0 0 0 0 0 0]
    #           [0 0 0 0 0 0]
    #           [0 3 0 0 4 0]
    #           [0 0 0 0 0 0]
    s = params.get('splat_interleave', 1)
    if s > 1:
        assert s % 2 == 1, 'splat_interleave must be odd'
        tmp = np.zeros((prob.shape[0], prob.shape[1] * s, prob.shape[2] * s), dtype=prob.dtype)
        tmp[:, (s-1)/2::s, (s-1)/2::s] = prob
        prob = tmp

    # bilinear (triangle filter) splatting
    if splat_triangle_weight > 0:
        for l in xrange(nlabels):
            _splat_prob_clip(
                target, (l, y0, x0, y1, x1),
                splat_triangle_weight *
                transform.resize(prob[l, :, :], foot_shape, order=1)
            )

    # nearest neighbor (box filter) splatting
    if splat_nearest_weight > 0:
        for l in xrange(nlabels):
            _splat_prob_clip(
                target, (l, y0, x0, y1, x1),
                splat_nearest_weight *
                transform.resize(prob[l, :, :], foot_shape, order=0)
            )

    # dirac delta splatting
    if splat_dirac_weight > 0:
        # be careful: prob and target are [C x H x W], foot is [H x W]
        for py in xrange(prob.shape[1]):
            uy = int(y0 + (py + 0.5) * foot_shape[0] / prob.shape[1] - 0.5)
            if uy < 0 or uy >= target.shape[1]:
                continue
            for px in xrange(prob.shape[2]):
                ux = int(x0 + (px + 0.5) * foot_shape[1] / prob.shape[2] - 0.5)
                if ux < 0 or ux >= target.shape[2]:
                    continue
                target[:, uy, ux] += (
                    splat_dirac_weight *
                    prob[:, py, px]
                )


def _splat_prob_clip(target, (l, y0, x0, y1, x1), source):
    assert source.shape == (y1 - y0, x1 - x0)
    _, height, width = target.shape
    if y0 < 0:
        source = source[(0 - y0):, :]
        y0 = 0
    if x0 < 0:
        source = source[:, (0 - x0):]
        x0 = 0
    if y1 > height:
        source = source[:(height - y1), :]
        y1 = height
    if x1 > width:
        source = source[:, :(width - x1)]
        x1 = width
    target[l, y0:y1, x0:x1] += source


def prob_footprints_to_unary(prob_footprints, prediction_shape, params):
    """ Splat prob_footprints onto a unary probability map """
    nlabels = prob_footprints[0]['prob'].shape[0]
    pred = np.zeros((nlabels, prediction_shape[0], prediction_shape[1]), dtype=np.float16)
    for p in prob_footprints:
        splat_prob(p['prob'], p['foot'], pred, params)
    pred /= len(prob_footprints)
    return pred


def serialize_prob_footprints(buf, prob_footprints):
    d = {}
    for i, p in enumerate(prob_footprints):
        d['%d-f' % i] = np.asarray(p['foot']).astype(np.float32)
        d['%d-p' % i] = (np.asarray(p['prob']) * 65535.0).astype(np.uint16)
    np.savez(buf, **d)


def deserialize_prob_footprints(buf):
    d = np.load(buf)
    foot = {}
    prob = {}
    max_id = 0
    for k, v in d.iteritems():
        id, kind = k.split('-')
        id = int(id)
        if kind == 'f':
            foot[id] = tuple(v)
        elif kind == 'p':
            prob[id] = v.astype(np.float32) / np.float32(65535.0)
        else:
            raise ValueError("Invalid kind: %s" % k)
        max_id = max(id, max_id)

    return [
        {'foot': foot[i], 'prob': prob[i]}
        for i in xrange(max_id+1)
    ]


def resize_unary(unary, new_shape, order=1):
    target = np.zeros((unary.shape[0], new_shape[0], new_shape[1]), unary.dtype)
    for l in xrange(unary.shape[0]):
        target[l, :, :] = transform.resize(unary[l, :, :], new_shape, order=order)
    return target


def predict_unary(image, prediction_shape, net_id, net=None):
    """ net: optionally pass in the net that was already loaded """

    params = config.NETWORK_CONFIGS[net_id]
    if not net:
        net = load_net(params)

    prob_footprints = forward_prob_footprints(image, net, params)
    unary = prob_footprints_to_unary(prob_footprints, prediction_shape, params)
    return unary
