import os
import csv
from datetime import datetime
import argparse
from tqdm import tqdm

from utils.logger import Logger
from src.dot_product import dot_product_cross_entropy_loss
from src.input_pipeline import InputPipeline
from model.model_manager import save_model, load_model, get_input_size

import pandas as pd
import numpy as np
import tensorflow as tf


logger = Logger()
logger.log('Start make_dot.py')


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--train_data', type=str, help='Path to train dataset csv file', 
                    default='data/train.csv')
parser.add_argument('-v', '--valid_data', type=str, help='Path to valid dataset csv file', 
                    default='data/valid.csv')
parser.add_argument('-t', '--test_data', type=str, help='Path to test dataset csv file', 
                    default='data/test.csv')
parser.add_argument('-i', '--input_dir', type=str, help='Path to the image directory', 
                    default='data/image')
parser.add_argument('-mc', '--model_config', help='Path to the model configuration json file', 
                    default=None)
parser.add_argument('-mw', '--model_weights', help='Path to the model weights file', 
                    default=None)
parser.add_argument('-e', '--epoch', type=int, default=30)
parser.add_argument('-tb', '--train_batch_size', type=int, default=300)
parser.add_argument('-vb', '--valid_batch_size', type=int, default=300)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
parser.add_argument('-sd', '--decay',  type=float, default=0.5)
parser.add_argument('-sp', '--patience',  type=int, default=5)
parser.add_argument('-g', '--clipping',  type=float, default=5.0)
parser.add_argument('-o', '--optimizer',  type=str, default='ADAM', choices=['ADAM', 'SGD'], 
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

valid_df = pd.concat(
        [pd.read_csv(args.valid_data), pd.read_csv(args.test_data)], axis=0, ignore_index=True
    ).sample(frac=1).reset_index()
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
    m = 'dot_inception_v32'
    base = tf.keras.applications.InceptionV3(
        include_top=False, 
        weights='imagenet',
        pooling='max',
    )
    base.trainable = True
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(1000),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
    ], name=m)

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
log_dir = f'./log/{model.name}/' + model.name + '_' + dt


# Define hyperparamenters
img_size = get_input_size(model.name)

# Prepare dataset
logger.log('Prepare dataset')
pipeline = InputPipeline(img_size=img_size)

train_batch_size = args.train_batch_size
train_ds = pipeline.load_dataset(
    train_df['filename'], train_df['make_id'], batch_size=train_batch_size, mode='train'
)
TRAIN_STEPS = len(train_df) // train_batch_size

valid_batch_size = args.valid_batch_size
valid_ds = pipeline.load_dataset(
    valid_df['filename'], valid_df['make_id'], batch_size=valid_batch_size, mode='valid'
)
VALID_STEPS = len(valid_df) // valid_batch_size


# Train
logger.log('Train model')

epochs = args.epoch
alpha = args.learning_rate
clipping = args.clipping
decay = args.decay
patience = args.patience

if args.optimizer.upper() == 'ADAM':
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
elif args.optimizer.upper() == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=0.9)
else:
    raise Exception('Invalid optimizer')
logger.log("starting learning rate:" + str(optimizer.lr.numpy()))


def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        emb = model(images, training=True)
        loss = dot_product_cross_entropy_loss(labels, emb)
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clipping)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test_step(model, images, labels):
    emb = model(images, training=False)
    loss = dot_product_cross_entropy_loss(labels, emb)
    return loss

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

training_loss = []
testing_loss = []
best_test_loss = None
patience_counter = 0
EPOCHS = args.epoch

pbar = tqdm(valid_ds.take(VALID_STEPS), total=VALID_STEPS, desc=f'valid 0')
for images, labels in pbar:
    loss = test_step(model, images, labels)
    test_loss(loss)
    pbar.set_postfix({
        'loss': str(test_loss.result().numpy()),
    })

with open(log_dir, 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow([
        'epoch', 'lr', 'train_loss', 'test_loss']
    )
    csv_writer.writerow([
        0, np.nan, np.nan, test_loss.result().numpy()
    ])

for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()

    pbar = tqdm(train_ds.take(TRAIN_STEPS), total=TRAIN_STEPS, desc=f'train {epoch + 1}')
    for images, labels in pbar:
        loss = train_step(model, images, labels)
        train_loss(loss)
        pbar.set_postfix({
            'loss': str(train_loss.result().numpy())
        })
    
    pbar = tqdm(valid_ds.take(VALID_STEPS), total=VALID_STEPS, desc=f'valid {epoch + 1}')
    for images, labels in pbar:
        loss = test_step(model, images, labels)
        test_loss(loss)
        pbar.set_postfix({
            'loss': str(test_loss.result().numpy()),
        })

    training_loss.append(train_loss.result().numpy())
    testing_loss.append(test_loss.result().numpy())

    with open(log_dir, 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow([
            epoch + 1, optimizer.lr.numpy(), training_loss[-1], testing_loss[-1]
        ])

    cur_loss = testing_loss[-1]

    if not best_test_loss or cur_loss < best_test_loss:
        model.save_weights(model_dir)
        best_test_loss = cur_loss
    else:
        patience_counter += 1

    # learning rate scheduler
    if patience_counter >= patience:
        alpha = alpha * decay
        tf.keras.backend.set_value(optimizer.lr, alpha)
        patience_counter = 0
        # early stopping
        if cur_loss > 0.2 + best_test_loss:
            break

logger.end('Done make_dot.py')
