#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import os
os.environ['GLOG_minloglevel']='2'
import caffe

def netsurgery(protofile, paramfile, params, protofile_full_conv, params_full_conv, opath):
    '''
    See https://github.com/BVLC/caffe/blob/master/examples/net_surgery.ipynb for
    details.

    protofile: prototxt of fully-connected model (read-only pathname)
    paramfile: weights of fully-connected model (read-only pathname)
    params: list of fully-connected blob names (list of str)
    protofile_full_conv: prototxt of fully-convolutional model (read-only pathname)
    params_full_conv: list of fully-convolutional blob names (list of str)
    opath: weights of fully-connected model (write pathname)
    '''
    caffe.set_mode_cpu()
    net = caffe.Net(protofile, paramfile, caffe.TEST)
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    for fc in params:
        print('{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape))

    net_full_conv = caffe.Net(protofile_full_conv, paramfile, caffe.TEST)
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
    for conv in params_full_conv:
        print('{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape))

    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    net_full_conv.save(opath)

if __name__=='__main__':
    netsurgery('minc-model/deploy-googlenet.prototxt', 
               'minc-model/minc-googlenet.caffemodel',
               ['fc8-20'],
               'deploy-googlenet_full_conv_no_pooling.prototxt',
               ['fc8-20-conv'],
               'minc-googlenet_full_conv.caffemodel')

