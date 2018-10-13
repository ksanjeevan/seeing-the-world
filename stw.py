#!/usr/bin/env python3

import argparse, json, os
from keras import backend as K

from net import Net


argparser = argparse.ArgumentParser(description='Image recognition.')


argparser.add_argument(
    'mode',
    help='what to do'
)
argparser.add_argument(
    '-w',
    '--weights',
    help='path to keras weights for inference/validation.',
    default=''
)

argparser.add_argument(
    '-c',
    '--conf',
    help='path to config file',
    default='config.json'
)


args = argparser.parse_args()

mode = args.mode
weight_path = args.weights
conf = args.conf


with open(conf, 'r') as r:
    config = json.load(r)


if weight_path and os.path.isfile(weight_path):
    config['trained_model_weights'] = weight_path


net = Net(config)

if mode == 'train':
    # Train network
    net.train()
elif mode == 'test_gen':
    # Test generator
    net.test_generator()
else:
    # Run inference on image
    net.infer(mode)

K.clear_session()