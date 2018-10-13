
import os, cv2
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from datagen import DataGenerator
from utils import setup_logging, mkdir_p


class TrainValTensorBoard(TensorBoard):
    """
    Tensorboard with Validation metrics
    """
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



def format_training_confs(config):
    """
    Set up configs for training/validation.
    """
    t_config = {
                'batch_size'    :   config['train']['batch_size'],
                'num_per_class' :   config['train']['num_per_class'][0],
                'num_classes'   :   config['train']['num_classes'],
                'size'          :   config['size']
    }


    v_config = {
                'batch_size'    :   config['train']['batch_size'],
                'num_per_class' :   config['train']['num_per_class'][1],
                'num_classes'   :   config['train']['num_classes'],
                'size'          :   config['size']
    }

    return (t_config, v_config)


def setup_gens(data, confs):
    """
    Create training/validation generators.
    """
    t_config, v_config = confs

    train_data = [d for d in data if d['train']==True]
    val_data = [d for d in data if d['train']==False]

    train_gen = DataGenerator(train_data, t_config)
    val_gen = DataGenerator(val_data, v_config)

    return train_gen, val_gen


def training(data, model, config):
    # Setp up generators
    confs = format_training_confs(config)    
    train_gen, val_gen = setup_gens(data, confs)
    
    # Set up logging/checkpointing dirs
    log_path = setup_logging()

    early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=5,
            mode='min',
            verbose=1)

    checkpoint_path = os.path.join(log_path, 'trained_weights.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1,
        save_weights_only=True
        )

    tensorboard = TrainValTensorBoard(log_dir=log_path)
    opt = Adam(lr=config['train']['lr'])

    # Use cross entropy for sparse labels
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
        epochs=config['train']['epochs'],
        callbacks=[checkpoint, tensorboard, early_stop])

    
def test_generator(data, config):
    # Create dir to store test images
    path = os.path.join(os.getcwd(), 'test_generator')
    mkdir_p(path)

    # Create generators
    confs = format_training_confs(config)    
    train_gen, val_gen = setup_gens(data, confs)

    # Pull small amount of images to check on augmentation 
    count = 0
    for i in range(3):
        images,_ = train_gen[i]
        for im in images:
            out = os.path.join(path, 'im_%s.png'%count)
            cv2.imwrite(out, cv2.resize(im, (0,0), fx=2., fy=2.))
            count += 1