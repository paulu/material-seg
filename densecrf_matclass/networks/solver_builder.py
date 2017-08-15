import os
from caffe.proto import caffe_pb2
from google.protobuf import text_format

SolverMode = caffe_pb2.SolverParameter.SolverMode


def build_solver(solver_filename, **kwargs):
    solver = caffe_pb2.SolverParameter()
    for k, v in kwargs.iteritems():
        setattr(solver, v)
    with open(solver_filename, 'w') as f:
        f.write(text_format.MessageToString(solver))


def build_solver_spearmint(
        solver_filename, network_filename, base_lr, max_iter, test_iter,
        gpu=True, momentum=0.9, weight_decay=0.0005):

    if gpu:
        solver_mode = SolverMode.Value('GPU')
    else:
        solver_mode = SolverMode.Value('CPU')

    build_solver(
        net=network_filename,
        snapshot_prefix=os.path.splitext(network_filename)[0] + '_',
        base_lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        test_iter=test_iter,
        test_interval=max_iter,
        max_iter=max_iter,
        test_initialization=False,
        snapshot_after_train=False,
        snapshot=0,
        display=0,
        solver_mode=solver_mode,
        stepsize=max_iter,
        lr_policy="step",
        gamma=1,
        random_seed=1337,
    )


# Example solver for alexnet:
#net: "models/bvlc_alexnet/train_val.prototxt"
#test_iter: 1000
#test_interval: 1000
#base_lr: 0.01
#lr_policy: "step"
#gamma: 0.1
#stepsize: 100000
#display: 20
#max_iter: 450000
#momentum: 0.9
#weight_decay: 0.0005
#snapshot: 10000
#snapshot_prefix: "models/bvlc_alexnet/caffe_alexnet_train"
#solver_mode: GPU
