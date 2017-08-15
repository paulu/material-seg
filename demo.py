#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import argparse
import sys
import subprocess
import os
os.environ['GLOG_minloglevel']='2'
import os.path
import caffe
import scipy.ndimage.interpolation
from matclass import dataset
import densecrf_matclass.general_densecrf
import imageutils

def labels_to_color(labels):
    """ Convert netcat labels to a color-mapped image """
    assert (labels <= dataset.UNKNOWN_LABEL).all() and (labels >= 0).all()
    image = numpy.zeros((labels.shape[0], labels.shape[1], 3), dtype=numpy.uint8)
    image[:, :, :] = dataset.LABEL_COLORS[labels]
    return image/255.0

def preprocess_and_reshape(data,model,blob='data',pre_scale=255.0,bgr_mean=numpy.array([104,117,124],dtype=numpy.float32),post_scale=1.0):
  '''
  This function follows the preprocessing steps of the Caffe Transformer
  except it resizes the model (if needed) instead of resizing the
  input. Returns the preprocessed data as a new array.

  The last three axes of data are height x width x rgb. Values are in
  the range [0,1]. Data is a rank-3 or rank-4 tensor.
  '''
  data=numpy.asarray(data,dtype=numpy.float32) # convert to single
  if data.ndim==3:
    data=data.transpose(2,0,1) # to channel x height x width
    data=data[numpy.newaxis,::-1,:,:] # RGB to BGR
  elif data.ndim==4:
    data=data.transpose(0,3,1,2) # to index x channel x height x width
    data=data[::-1,:,:] # RGB to BGR
  else:
    raise ValueError('preprocess() expects a rank-3 or rank-4 tensor.')
  data*=pre_scale # scale raw input
  data-=bgr_mean.reshape(3,1,1) # subtract training mean
  data*=post_scale # scale mean-subtracted input
  if list(model.blobs[blob].shape)!=list(data.shape):
    model.blobs[blob].reshape(*data.shape)
  return data

def nearest_multiple(x,n):
  'Returns the nearest multiple of n to x.'
  return int(round(x/float(n)))*n

def main(config):
  caffe.set_device(config['device_id'])
  caffe.set_mode_gpu()
  if config['arch']=='A4,G1':
    # an ensemble of alexnet and googlenet
    models=[
      caffe.Net('deploy-alexnet_full_conv.prototxt','minc-alexnet_full_conv.caffemodel',caffe.TEST),
      caffe.Net('deploy-googlenet_full_conv_no_pooling.prototxt','minc-googlenet_full_conv.caffemodel',caffe.TEST)
    ]

    # alexnet needs a padded input to get a full-frame prediction
    input_padding=[97,0]

    # nominal footprint is 46.4% for A4, 23.2% for G1
    scales=[256/550.0,256/1100.0]
    bgr_mean=numpy.array([104,117,124],dtype=numpy.float32)

    # inputs must be a multiple of the stride
    # otherwise, the full-frame prediction will be shifted
    effective_stride=[32,32]

    # TODO: A4 needs spatial oversampling (# shifts = 2)

    # these are the CRF parameters for MINC
    # the parameters can have a big impact on the output
    # so they should be tuned for the target domain
    crf_params={
      "bilateral_pairwise_weight": 5.0, # w_p
      "bilateral_theta_xy": 0.1, # \theta_p
      "bilateral_theta_lab_l": 20.0, # \theta_L
      "bilateral_theta_lab_ab": 5.0, # \theta_ab
      "min_dim": 550, # map size
      "n_crf_iters": 10,
      "splat_triangle_weight": 1,
      "unary_prob_padding": 1e-05,
      "ignore_labels": [dataset.NAME_TO_NETCAT['other']],
      "stride": 32,
    }
  else:
    raise NotImplementedError

  pad_value=bgr_mean[::-1]/255.0

  for ipath in config['input']:
    # read image
    original=imageutils.read(ipath)
    #print(ipath,original.shape)
    z=crf_params['min_dim']/float(min(original.shape[:2]))
    crf_shape=(23,int(round(original.shape[0]*z)),int(round(original.shape[1]*z)))

    # predict 6 maps: 3 scales for each model
    maps=[]
    for index,model in enumerate(models):
      p=input_padding[index]
      s=scales[index]
      for index2,multiscale in enumerate([0.7071067811865476,1.0,1.4142135623730951]):
        # resample the input so it is a multiple of the stride
        # and the receptive field matches the nominal footprint
        scale_factor=(256/s)/float(min(original.shape[:2]))
        scaled_size=[nearest_multiple(original.shape[i]*scale_factor*multiscale,effective_stride[index]) for i in range(2)]
        scaled=imageutils.resize(original,scaled_size)
        if p>0:
          # add input padding for alexnet
          pad=numpy.ones((scaled.shape[0]+2*p,scaled.shape[1]+2*p,scaled.shape[2]),dtype=scaled.dtype)*pad_value
          pad[p:-p,p:-p]=scaled
          scaled=pad

        # predict and resample the map to be the correct size
        data=preprocess_and_reshape(scaled,model,bgr_mean=bgr_mean)
        #print(index,index2,'data',data.shape)
        output=model.forward_all(data=data)['prob'][0]
        #print(index,index2,'output',output.shape)
        output=scipy.ndimage.interpolation.zoom(output,[1.0,crf_shape[1]/float(output.shape[1]),crf_shape[2]/float(output.shape[2])],order=1)
        maps.append(output)

    # average all maps
    crf_map=numpy.asarray(maps).mean(axis=0)
    #print(index,index2,'crf map',crf_map.shape)
    crf_color=imageutils.resize(original,crf_shape)
    lcrf=densecrf_matclass.general_densecrf.LearnableDenseCRF(crf_color,crf_map,crf_params)
    labels_crf=lcrf.map(crf_params)
    result=labels_to_color(labels_crf)
    imageutils.write('{}-result.jpg'.format(os.path.splitext(ipath)[0]),result)

def available_disk_space():
  st=os.statvfs('.')
  return st.f_bavail*st.f_frsize

if __name__=='__main__':
  # configure by command-line arguments
  parser=argparse.ArgumentParser(description='MINC full-scene material segmentation.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('input',type=str,nargs='+',help='input color image')
  parser.add_argument('--arch',type=str,default='A4,G1',choices=['googlenet','vgg16','alexnet'],help='pre-trained model architecture')
  parser.add_argument('--device_id',type=int,default=0,help='zero-indexed CUDA device')
  config=parser.parse_args()
  print('config',config.__dict__)

  if not os.path.exists('minc-model'):
    print('Downloading MINC models from http://minc.cs.cornell.edu/ and unpacking them ...')
    if available_disk_space()<2**30:
      print('WARNING: There is less than 2 GB of available space on this filesystem. If subsequent operations fail then either free up space or unpack the models elsewhere and create a symlink here.')
    subprocess.check_call(['wget','http://opensurfaces.cs.cornell.edu/static/minc/minc-model.tar.gz'])
    subprocess.check_call(['tar','xzvf','minc-model.tar.gz'])
    os.unlink('minc-model.tar.gz')

  if not os.path.exists('minc-googlenet_full_conv.caffemodel'):
    import netsurgery
    print('Applying net surgery ...')
    netsurgery.netsurgery('minc-model/deploy-googlenet.prototxt','minc-model/minc-googlenet.caffemodel',['fc8-20'],'deploy-googlenet_full_conv_no_pooling.prototxt',['fc8-20-conv'],'minc-googlenet_full_conv.caffemodel')

  if not os.path.exists('minc-alexnet_full_conv.caffemodel'):
    import netsurgery
    print('Applying net surgery ...')
    netsurgery.netsurgery('minc-model/deploy-alexnet.prototxt','minc-model/minc-alexnet.caffemodel',['fc6','fc7','fc8-20'],'deploy-alexnet_full_conv.prototxt',['fc6-conv','fc7-conv','fc8-20-conv'],'minc-alexnet_full_conv.caffemodel')
    #227 54 25 11 4

  #main({'input':['images/0013.jpg'], 'arch':'A4,G1', 'device_id':0})
  main(config.__dict__)

