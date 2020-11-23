import os
from datetime import datetime
import argparse

from utils.logger import Logger
from src.input_pipeline import InputPipeline
from model.model_manager import save_model, load_model, get_input_size
# from model.googlenet.base.googlenet import Googlenet

import pandas as pd
import numpy as np
import tensorflow as tf


logger = Logger()
logger.log('Start model_train.py')


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--train_data', help='Path to train dataset csv file', default='data/data.csv')
parser.add_argument('-v', '--valid_data', help='Path to valid dataset csv file', default=None)
parser.add_argument('-i', '--input_dir', help='Path to the image directory', default='data/image')
parser.add_argument('-mc', '--model_config', help='Path to the model configuration json file', default=None)
parser.add_argument('-mw', '--model_weights', help='Path to the model weights file', default=None)
parser.add_argument('-e', '--epoch', type=int, default=50)
parser.add_argument('-tb', '--train_batch_size', type=int, default=200)
parser.add_argument('-vb', '--valid_batch_size', type=int, default=500)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
parser.add_argument('-clr', '--change_learning_rate', default=None)
parser.add_argument('-s', '--lr_scheduler', type=eval, choices=[True, False], default='False')
parser.add_argument('-o', '--optimizer',  type=str, default='ADAM', choices=['ADAM', 'SGD', 'RMSPROP'], 
                    help='Type of optimizer (ADAM or SGD)')
args = parser.parse_args()


# Load dataset
logger.log('Load dataset')
input_dir = args.input_dir if args.input_dir[-1] == '/' else args.input_dir + '/'

train_df = pd.read_csv(args.train_data).sample(frac=1).reset_index()
train_df['filename'] = input_dir + \
                 train_df['make_id'].astype(str) + '/' + \
                 train_df['model_id'].astype(str) + '/' + \
                 train_df['released_year'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'unknown') + '/' + \
                 train_df['image_name']

if args.valid_data:
    valid_df = pd.read_csv(args.valid_data)
    valid_df['filename'] = input_dir + \
                valid_df['make_id'].astype(str) + '/' + \
                valid_df['model_id'].astype(str) + '/' + \
                valid_df['released_year'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'unknown') + '/' + \
                valid_df['image_name']


# Load model
logger.log('Load model')
if args.model_config:
    model = load_model(args.model_config, args.model_weights)
else:
    pass
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False, 
        weights='imagenet',
        pooling='max'
    )
    inception_v3.trainable = True
    model = tf.keras.Sequential([
        inception_v3,
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(2000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=1e-3)),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(2000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=1e-3)),
        tf.keras.layers.Dense(2004, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=1e-3))
    ], name='inception_v3_model_classifier9')


# Configure directory
dt = datetime.now().strftime("%Y%m%d%H%M%S")

if not os.path.exists('./model'):
    os.mkdir('./model')
if not os.path.exists(f'./model/{model.name}'):
    os.mkdir(f'./model/{model.name}')
model_dir = f'./model/{model.name}/' + model.name + '_' + dt
save_model(model, model_dir + '.json')

if not os.path.exists('./log'):
    os.mkdir('./log')
if not os.path.exists(f'./log/{model.name}'):
    os.mkdir(f'./log/{model.name}')
log_dir = f'./log/{model.name}/' + model.name + '_' + dt + '_train.log'


# Define hyperparamenters
img_size = get_input_size(model.name)
epochs = args.epoch
alpha = args.learning_rate


# Prepare dataset
logger.log('Prepare dataset')
pipeline = InputPipeline(img_size=img_size)

train_batch_size = args.train_batch_size
train_ds = pipeline.load_dataset(train_df['filename'], train_df['model_id'], batch_size=train_batch_size, mode='train')
TRAIN_STEPS = len(train_df) // train_batch_size

if args.valid_data:
    valid_batch_size = args.valid_batch_size
    valid_ds = pipeline.load_dataset(valid_df['filename'], valid_df['model_id'], batch_size=valid_batch_size, mode='valid')
    VALID_STEPS = len(valid_df) // valid_batch_size
else:
    valid_ds = None
    VALID_STEPS = None


# Train
logger.log('Train model')

def make_callbacks():
    callbacks = list()

    if args.valid_data:
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(
            model_dir, 
            monitor='val_loss', 
            save_best_only=True,
            save_weights_only=True
        )
    else:
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(
            model_dir, 
            monitor='loss', 
            save_best_only=True,
            save_weights_only=True
        )
    callbacks.append(model_ckpt)

    if args.lr_scheduler:
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_delta=0.01
        )
        callbacks.append(lr_scheduler)

    update_history = tf.keras.callbacks.CSVLogger(
        log_dir, separator=',', append=True
    )
    callbacks.append(update_history)
    return callbacks

callbacks = make_callbacks()

if args.optimizer.upper() == 'ADAM':
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
elif args.optimizer.upper() == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=0.9)
elif args.optimizer.upper() == 'RMSPROP':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=0.9, momentum=0.9, epsilon=1.0)
else:
    raise Exception('Invalid optimizer')

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=1, name='top1_accuracy', dtype=None
        ),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5, name='top5_accuracy', dtype=None
        )
    ]
)

if args.change_learning_rate:
    tf.keras.backend.set_value(model.optimizer.lr, float(args.change_learning_rate))
logger.log("starting learning rate:" + str(model.optimizer.lr.numpy()))

hist = model.fit(
    train_ds,
    epochs=epochs,
    verbose=1,
    validation_data=valid_ds,
    steps_per_epoch=TRAIN_STEPS,
    validation_steps=VALID_STEPS,
    callbacks=callbacks
)

logger.end('Done model_train.py')
