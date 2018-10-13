
import imgaug as ia
from imgaug import augmenters as iaa

import os,cv2
from keras.utils import Sequence
import numpy as np

np.warnings.filterwarnings('ignore')
ia.seed(1)


def parse_input_data(path, split=0.75):
    """
    Parse training images for DataGenerator to process.
    """
    ret = []
    class_map = {}

    # Get all classes as directory names
    classes = [l for l in os.listdir(path) if not l.startswith('.')]


    for k, c in enumerate(classes):
        c_path = os.path.join(path, c)

        # Get imaeges in folder
        image_names = [l for l in os.listdir(c_path) if not l.startswith('.')]
        np.random.shuffle(image_names)

        # Tag certain % of them as train/validation
        M = len(image_names)
        N = int(round(split*M))
        training = [True]*N + [False]*(M-N)

        for j, im_name in enumerate(image_names):
            full_path = os.path.join(c_path, im_name)
            ret.append( {
                'filename':full_path, 
                'class':c, 
                'class_index':k,
                'train':training[j]
                } )
        # Store decoder for label names
        class_map[k] = c

    return ret, class_map



class DataGenerator(Sequence):

    def __init__(self, train_ims, config, shuffle=True):

        # Network parameters
        self.H, self.W = config['size']
        self.num_classes = config['num_classes']
        self.num_per_class = config['num_per_class']
        self.batch_size = config['batch_size']
        
        self.shuffle = shuffle
        
        # Create copies of original dataset to satisfy desired
        # number of images per class. Due to the randomized 
        # augmentation this should provide different examples.
        self.train_ims = self.preprocess_images(train_ims)

        # Augmenter class to wrap imgaug
        self.augmenter = DataAugmenter()
        
        self.on_epoch_end()


    def __len__(self):
        """
        Denotes number of batches per epoch.
        """
        return int(round(self.num_classes*self.num_per_class/self.batch_size))


    def __getitem__(self, i):
        """
        Generator output.
        """
        start = self.batch_size*i
        end = self.batch_size*(i+1)

        return self.get_data(self.train_ims[start:end])


    def preprocess_images(self, ims):
        ret = []
        
        # Get images for each class
        per_class = []
        for c in range(self.num_classes):
            per_class.append([im for im in ims if im['class_index'] == c])

        # Make copies up to num_per_class
        for ims in per_class:
            count = 0
            index = 0
            while count < self.num_per_class:                
                ret.append( ims[index] )
                if index >= len(ims)-1:
                    index = 0
                else:
                    index += 1
                count += 1

        return ret


    def get_data(self, names):

        # Read in images and resize them for network
        images = [cv2.imread(n['filename']) for n in names]
        images = self.resize_images(images)

        # Read in label array
        labels = np.array([n['class_index'] for n in names])

        # Run augmentation on images
        return self.augmenter.run_aug(images), labels

    def resize_images(self, images):
        """
        H,W resizing.
        """
        return [cv2.resize(im, (self.H, self.W)) for im in images]

    def on_epoch_end(self):
        """
        Shuffle images on epoch end.
        """
        if self.shuffle: np.random.shuffle(self.train_ims)



class DataAugmenter(object):
    """
    imgaug wrapper.
    """
    def __init__(self):
        pass

    def run_aug(self, images):

        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.15)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.55, 1.8)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.7),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.5, 1.5), per_channel=0.4),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-35, 35),
                shear=(-12, 12)
            )
            ], random_order=True) # apply augmenters in random order


        return np.array(seq.augment_images(images))




if __name__ == '__main__':

    ims, cmap = parse_input_data('data/farmer_market')

    config = {'batch_size':16, 'num_per_class':30, 'num_classes':12}

    g = DataGenerator(ims, config)
    ims, labels = g[0]

    
