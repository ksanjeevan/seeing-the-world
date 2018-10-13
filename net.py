

from keras.applications import VGG16
from keras.models import Model, load_model
from keras.layers import Dense
import numpy as np

from train import training

import cv2

from datagen import parse_input_data

def update_model(model, out_size):

    new_out = Dense(
        out_size, 
        activation='softmax', 
        name='predictions')(model.layers[-2].output)

    return Model(inputs=model.inputs, outputs=new_out)


class Net(object):

    def __init__(self, config):
        
        self.model = VGG16()
        self.config = config

        self.H, self.W = config['size']
        self.data, self.index_to_class = parse_input_data(
            config['path'], 
            config['train']['split'])

        self.num_classes = len(self.index_to_class)

        self.config['train']['num_classes'] = self.num_classes

        if 'trained_model' in config:
            self.t_model = load_model(config['trained_model'])


    def train(self):
        new_model = update_model(self.model, self.num_classes)

        training(self.data, new_model, self.config)


    def infer(self, path_image):

        # Pre-processing image
        image = cv2.imread(path_image)
        image = cv2.resize(image, (self.H, self.W))
        iamge = np.expand_dims(image, axis=0)        

        out = self.t_model.predict(iamge)[0]

        winner_ind = np.argmax(out)
        winner_conf = out[winner_ind]

        winner_label = self.index_to_class[winner_ind]

        print("Prediction: %s (%.2f%%)"%(winner_label, winner_conf*100.))

