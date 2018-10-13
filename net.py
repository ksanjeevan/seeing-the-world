
import numpy as np
import cv2

from keras.applications import VGG16
from keras.models import Model, load_model
from keras.layers import Dense

from train import training, test_generator
from datagen import parse_input_data


def update_model(model, out_size):

    new_out = Dense(
        out_size, 
        activation='softmax', 
        name='predictions')(model.layers[-2].output)

    return Model(inputs=model.inputs, outputs=new_out)


class Net(object):
    """
    Handle training and inference of the network. 
    Training makes use of Imagenet transfer learning and 
    randomizes last layer (no freezing). 
    """
    def __init__(self, config):
        # Set up clasification network
        self.model = VGG16()
        self.config = config

        # Parse data and label decoder
        self.data, self.index_to_class = parse_input_data(
            config['path'], 
            config['train']['split'])

        # Store network parameters
        self.H, self.W = config['size']
        self.num_classes = len(self.index_to_class)
        self.config['train']['num_classes'] = self.num_classes

        # Randomized last layer with approiate output size
        self.trained_model = update_model(self.model, self.num_classes)

        if 'trained_model_weights' in config:
            
            self.trained_model.load_weights(config['trained_model_weights'])


    def train(self):
        # Perform training with config parameters
        training(self.data, self.trained_model, self.config)


    def test_generator(self):
        # Output small number of augmented training images
        # for testing
        test_generator(self.data, self.config)


    def infer(self, path_image):

        # Pre-processing image
        image = cv2.imread(path_image)
        image = cv2.resize(image, (self.H, self.W))
        iamge = np.expand_dims(image, axis=0)        

        # Get network prediction
        out = self.trained_model.predict(iamge)[0]

        # Pick class with most confidence
        winner_ind = np.argmax(out)
        winner_conf = out[winner_ind]

        # Decode label
        winner_label = self.index_to_class[winner_ind]

        print("Prediction: %s (%.2f%%)"%(winner_label, winner_conf*100.))

