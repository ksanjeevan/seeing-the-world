
import os
from datagen import parse_input_data, DataGenerator

from utils import setup_logging

import tensorflow as tf

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam


PATH = 'data/farmer_market'

t_config = {'batch_size':16, 'num_per_class':120, 'num_classes':12}
v_config = {'batch_size':16, 'num_per_class':30, 'num_classes':12}

NUM_CLASSES = 12
LR = 0.0002
EPOCHS = 100


class TrainValTensorBoard(TensorBoard):

    def __init__(self, log_dir='./logs', **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        self.validation_log_dir = os.path.join(log_dir, 'validation')


    def set_model(self, model):

        self.val_writer = tf.summary.FileWriter(self.validation_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_logs = {k.replace('val_', ''):v for k,v in logs.items() if k.startswith('val')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k:v for k,v in logs.items() if not k.startswith('val')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()



def training(model, confs):

    t_config, v_config = confs

    data, index_to_class = parse_input_data(PATH)

    train_data = [d for d in data if d['train']==True]
    val_data = [d for d in data if d['train']==False]

    train_gen = DataGenerator(train_data, t_config)
    val_gen = DataGenerator(val_data, v_config)


    log_path = setup_logging()    


    early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=5,
            mode='min',
            verbose=1)


    checkpoint_path = os.path.join(log_path, 'trained_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
        )

    tensorboard = TrainValTensorBoard(log_dir=log_path)

    opt = Adam(lr=LR)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt, 
        metrics=['acc'])



    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=len(train_gen),
        verbose=1,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=EPOCHS,
        callbacks=[checkpoint, tensorboard, early_stop])



    K.clear_session()


from keras.models import Model
from keras.layers import Dense


def update_model(model):

    new_out = Dense(
        NUM_CLASSES, 
        activation='softmax', 
        name='predictions')(model.layers[-2].output)

    return Model(inputs=model.inputs, outputs=new_out)


    
if __name__ == '__main__':

    from keras.applications import VGG16

    model = VGG16()

    new_model = update_model(model)

    confs = t_config, v_config

    training(new_model, confs)
