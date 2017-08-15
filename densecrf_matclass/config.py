import os
import numpy as np

from util import hex_to_rgb

NUM_PROCESSES = 8

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'models', 'matclass-scene', 'models'
)

DENSECRF_CONFIG = {
    'unary_prob_padding': 1e-6,
    'n_crf_iters': 5,
    #'bilateral_pairwise_weight': 1.0,
    #'bilateral_theta_lab_l': 1.0,
    #'bilateral_theta_lab_ab': 1.0,
    #'bilateral_theta_xy': 1.0,
    'log_likelihood_robust': 0.1,
    'min_dim': 1100,
    #'position_pairwise_weight': 0,
    #'position_theta_xy': 0.05,

    # best params on 32 validation images
    "bilateral_pairwise_weight": 1.2595191348106027,
    "bilateral_theta_lab_l": 1.1741073710553565,
    "bilateral_theta_xy": 1.0087199292016749,
    "bilateral_theta_lab_ab": 1.1531286310489577

    # best params with interlaced splatting
    #"bilateral_pairwise_weight": 1.266774237894424,
    #"bilateral_theta_lab_l": 1.0874884757067775,
    #"bilateral_theta_xy": 0.93931851780143916,
    #"bilateral_theta_lab_ab": 1.20344257101533,
}

# photos used to tarin the CRFs
#DENSECRF_TRAIN_PHOTO_IDS = [
    #int(f.strip()) for f in open(
        #os.path.join(DATA_DIR, 'densecrf_train_images.txt')
    #)
#]


# NETWORKS
NETWORK_CONFIGS = {
    ## scene
    #'003': {
        #'mode': 'grid',
        #'min_dim': 1100,
        #'caffemodel': os.path.join(MODEL_DIR, '003-iter202000.caffemodel'),
        #'deploy': os.path.join(MODEL_DIR, '003-deploy.prototxt'),
        #'has_global_data': True,

        #'mirror': False,
        #'oversample_x': 2,
        #'oversample_y': 2,
        #'oversample_scale': 3,
        #'min_scale': 1,
        #'max_scale': np.sqrt(2),

        #'splat_triangle_weight': 1,
    #},
    ## scene+densecrf
    #'003-densecrf': {
        #'mode': 'grid',
        #'min_dim': 1100,
        #'caffemodel': os.path.join(MODEL_DIR, '003-iter202000.caffemodel'),
        #'deploy': os.path.join(MODEL_DIR, '003-deploy-densecrf.prototxt'),
        #'has_global_data': True,

        #'mirror': False,
        #'oversample_x': 1,
        #'oversample_y': 1,
        #'oversample_scale': 1,
        #'min_scale': 1,
        #'max_scale': np.sqrt(2),

        #'splat_triangle_weight': 1,
    #},
    ## foot:1024 out:20x16x16
    #'102': {
        #'mode': 'grid',
        #'min_dim': 227 * 1100 / 1024,
        #'caffemodel': os.path.join(MODEL_DIR, '102-iter84000.caffemodel'),
        #'deploy': os.path.join(MODEL_DIR, '102-deploy.prototxt'),

        #'mirror': False,
        #'oversample_x': 2,
        #'oversample_y': 2,
        #'oversample_scale': 3,
        #'min_scale': 1,
        #'max_scale': np.sqrt(2),

        #'splat_triangle_weight': 1,
    #},
    ## foot:256 out:20x4x4
    #'110': {
        #'mode': 'grid',
        #'min_dim': 227 * 1100 / 256,
        #'caffemodel': os.path.join(MODEL_DIR, '110-iter54000.caffemodel'),
        #'deploy': os.path.join(MODEL_DIR, '110-deploy.prototxt'),
        #'oversample_x': 5,
        #'oversample_y': 5,
        #'mirror': True,

        #'splat_triangle_weight': 1,
    #},
    # patch256
    '201': {
        'mode': 'sliding',
        'min_dim': 256 * 1100 / 256,
        'caffemodel': os.path.join(MODEL_DIR, '201-iter204000.caffemodel'),
        'deploy_target': os.path.join(MODEL_DIR, '201-deploy-conv.prototxt'),
        'deploy_source': os.path.join(MODEL_DIR, '201-deploy.prototxt'),
        'param_mapping': {
            'fc6-conv': 'fc6',
            'fc7-conv': 'fc7',
            'fc8-conv': 'fc8-20',
        },
        'input_pad': 97,
        'effective_stride': 32,
        'receptive_field': 227,
        'oversample_scale': 3,
        'oversample_pad': 2,
        'mirror': False,

        #'splat_interleave': 3,
        'splat_triangle_weight': 1,
    },
    # patch352
    '202': {
        'mode': 'sliding',
        'min_dim': 256 * 1100 / 352,
        'caffemodel': os.path.join(MODEL_DIR, '202-iter24000.caffemodel'),
        'deploy_target': os.path.join(MODEL_DIR, '202-deploy-conv.prototxt'),
        'deploy_source': os.path.join(MODEL_DIR, '202-deploy.prototxt'),
        'param_mapping': {
            'fc6-conv': 'fc6',
            'fc7-conv': 'fc7',
            'fc8-conv': 'fc8-20',
        },
        'input_pad': 97,  # (227 - 1 - 32)/2 = 97
        'effective_stride': 32,
        'receptive_field': 227,
        'oversample_scale': 3,
        'oversample_pad': 2,
        'mirror': False,

        'splat_triangle_weight': 1,
    },
    ## patch128
    #'203': {
    #    'mode': 'sliding',
    #    'min_dim': 256 * 1100 / 128,
    #    'caffemodel': os.path.join(MODEL_DIR, '203-iter26000.caffemodel'),
    #    'deploy_target': os.path.join(MODEL_DIR, '203-deploy-conv.prototxt'),
    #    'deploy_source': os.path.join(MODEL_DIR, '203-deploy.prototxt'),
    #    'param_mapping': {
    #        'fc6-conv': 'fc6',
    #        'fc7-conv': 'fc7',
    #        'fc8-conv': 'fc8-20',
    #    },
    #    'input_pad': 97,
    #    'effective_stride': 32,
    #    'receptive_field': 227,
    #    'oversample_scale': 3,
    #    'oversample_pad': 1,
    #    'mirror': False,

    #    'splat_triangle_weight': 1,
    #},
    # VGG
    '209': {
        'mode': 'sliding',
        'min_dim': 256 * 1100 / 256,
        'caffemodel': os.path.join(MODEL_DIR, '209-iter210000.caffemodel'),
        'deploy_target': os.path.join(MODEL_DIR, '209-deploy-conv.prototxt'),
        'deploy_source': os.path.join(MODEL_DIR, '209-deploy.prototxt'),
        'param_mapping': {
            'fc6-conv': 'fc6',
            'fc7-conv': 'fc7',
            'fc8-conv': 'fc8-20',
        },
        'input_pad': 96,
        'effective_stride': 32,
        'receptive_field': 224,
        'oversample_scale': 3,
        'oversample_pad': 1,
        'mirror': False,

        #'splat_interleave': 3,
        'splat_triangle_weight': 1,
    },
}


# LABELS
LABEL_TO_NAME = [
    f.strip() for f in open(
        os.path.join(DATA_DIR, 'classify_categories.txt')
    )
]
NAME_TO_LABEL = {
    x: i for (i, x) in enumerate(LABEL_TO_NAME)
}
UNKNOWN_LABEL = NAME_TO_LABEL['unknown']
OTHER_LABEL = NAME_TO_LABEL['other']
NLABELS = 23

LABEL_COLORS = np.array([
    hex_to_rgb(h) for h in [
        '#771111', '#cac690', '#eeeeee', '#7c8fa6', '#597d31', '#104410',
        '#bb819c', '#d0ce48', '#622745', '#666666', '#d54a31', '#101044',
        '#444126', '#75d646', '#dd4348', '#5c8577', '#c78472', '#75d6d0',
        '#5b4586', '#c04393', '#d69948', '#7370d8', '#7a3622', '#000000',
    ]
], dtype=np.uint8)

assert len(LABEL_COLORS) == NLABELS + 1

#GROUND_SET_TO_LABEL = np.array([
    #int(f.strip()) for f in open(
        #os.path.join(DATA_DIR, 'ground_set_mapping.txt')
    #)
#])

# CLASS WEIGHTS
#CLASS_WEIGHTS = np.array([
    #1.0 / float(f.strip()) for f in open(
        #os.path.join(DATA_DIR, 'category_frequencies.txt')
    #)
#] + [0.0])

#CLASS_WEIGHTS /= np.max(CLASS_WEIGHTS)
