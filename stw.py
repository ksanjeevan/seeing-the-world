

import argparse, json, os

from net import Net


argparser = argparse.ArgumentParser(description='Image recognition.')


argparser.add_argument(
    'mode',
    help='what to do'
)
argparser.add_argument(
    '-m',
    '--model',
    help='path to keras model for inference',
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
model = args.model
conf = args.conf


with open(conf, 'r') as r:
    config = json.load(r)


if model and os.path.isfile(model):
    config['trained_model'] = model


net = Net(config)

if mode == 'train':
    net.train()    
else:
    net.infer(mode)
