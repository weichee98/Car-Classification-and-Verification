import os
import csv
from datetime import datetime
import argparse
from tqdm import tqdm

from utils.logger import Logger
from src.triplet import triplet_loss, triplet_loss_hard, TripletInput
from src.triplet import batch_all_triplet_loss
from src.input_pipeline import InputPipeline
from model.model_manager import save_model, load_model, get_input_size

import pandas as pd
import numpy as np
import tensorflow as tf


logger = Logger()
logger.log('Start make_triplet.py')


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
parser.add_argument('-m', '--margin', type=float, default=0.5)
parser.add_argument('-sd', '--decay',  type=float, default=0.5)
parser.add_argument('-sp', '--patience',  type=int, default=5)
parser.add_argument('-g', '--clipping',  type=float, default=5.0)
parser.add_argument('-o', '--optimizer',  type=str, default='ADAM', choices=['ADAM', 'SGD'], 
                    help='Type of optimizer (ADAM or SGD)')
parser.add_argument('-md', '--mode',  type=str, default='SEMIHARD', choices=['ALL', 'SEMIHARD', 'HARD'], 
                    help='Mode (ALL, SEMIHARD, HARD)')
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
    m = 'triplet_inception_v3_' + args.mode.lower()
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
train_ds = pipeline.load_balanced_dataset(
    train_df['filename'], train_df['make_id'], batch_size=train_batch_size, mode='train'
)
TRAIN_STEPS = len(train_df) // train_batch_size

valid_batch_size = args.valid_batch_size
valid_ds = pipeline.load_balanced_dataset(
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
margin = args.margin

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
        all_loss, hard, semihard = batch_all_triplet_loss(labels, emb, margin, semi_hard=True)
        if args.mode == 'HARD':
            loss = hard
        elif args.mode == 'SEMIHARD':
            loss = semihard
        else:
            loss = all_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clipping)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return all_loss, hard, semihard

def test_step(model, images, labels):
    emb = model(images, training=False)
    all_loss, hard, semihard = batch_all_triplet_loss(labels, emb, margin, semi_hard=True)
    return all_loss, hard, semihard

train_all = tf.keras.metrics.Mean(name='train_all')
train_hard = tf.keras.metrics.Mean(name='train_hard')
train_semihard = tf.keras.metrics.Mean(name='train_semihard')
test_all = tf.keras.metrics.Mean(name='test_all')
test_hard = tf.keras.metrics.Mean(name='test_hard')
test_semihard = tf.keras.metrics.Mean(name='test_semihard')

training_all = []
training_hard = []
training_semihard = []
testing_all = []
testing_hard = []
testing_semihard = []
best_all = None
best_hard = None
best_semihard = None
patience_counter = 0
EPOCHS = args.epoch

pbar = tqdm(valid_ds.take(VALID_STEPS), total=VALID_STEPS, desc=f'valid 0')
for images, labels in pbar:
    loss = test_step(model, images, labels)
    test_all(loss[0])
    test_hard(loss[1])
    test_semihard(loss[2])
    pbar.set_postfix({
        'all': str(test_all.result().numpy()),
        'hard': str(test_hard.result().numpy()),
        'semihard': str(test_semihard.result().numpy())
    })

with open(log_dir, 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow([
        'epoch', 'lr', 'train_all', 'train_hard', 'train_semihard', 
        'test_all', 'test_hard', 'test_semihard']
    )
    csv_writer.writerow([
        0, np.nan, np.nan, np.nan, np.nan,
        test_all.result().numpy(), test_hard.result().numpy(), test_semihard.result().numpy(), 
    ])

for epoch in range(EPOCHS):
    train_all.reset_states()
    train_hard.reset_states()
    train_semihard.reset_states()
    test_all.reset_states()
    test_hard.reset_states()
    test_semihard.reset_states()

    pbar = tqdm(train_ds.take(TRAIN_STEPS), total=TRAIN_STEPS, desc=f'train {epoch + 1}')
    for images, labels in pbar:
        loss = train_step(model, images, labels)
        train_all(loss[0])
        train_hard(loss[1])
        train_semihard(loss[2])
        pbar.set_postfix({
            'all': str(train_all.result().numpy()),
            'hard': str(train_hard.result().numpy()),
            'semihard': str(train_semihard.result().numpy())
        })
    
    pbar = tqdm(valid_ds.take(VALID_STEPS), total=VALID_STEPS, desc=f'valid {epoch + 1}')
    for images, labels in pbar:
        loss = test_step(model, images, labels)
        test_all(loss[0])
        test_hard(loss[1])
        test_semihard(loss[2])
        pbar.set_postfix({
            'all': str(test_all.result().numpy()),
            'hard': str(test_hard.result().numpy()),
            'semihard': str(test_semihard.result().numpy())
        })

    training_all.append(train_all.result().numpy())
    training_hard.append(train_hard.result().numpy())
    training_semihard.append(train_semihard.result().numpy())
    testing_all.append(test_all.result().numpy())
    testing_hard.append(test_hard.result().numpy())
    testing_semihard.append(test_semihard.result().numpy())

    with open(log_dir, 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow([
            epoch + 1, optimizer.lr.numpy(), 
            training_all[-1], training_hard[-1], training_semihard[-1], 
            testing_all[-1], testing_hard[-1], testing_semihard[-1], 
        ])

    cur_all = testing_all[-1]
    cur_hard = testing_hard[-1]
    cur_semihard = testing_semihard[-1]

    if not best_all and not best_hard and not best_semihard:
        model.save_weights(model_dir)
        best_all = cur_all
        best_hard = cur_hard
        best_semihard = cur_semihard
    else:
        update = False
        if cur_all <= best_all:
            best_all = cur_all
            update = True
        if cur_hard <= best_hard:
            best_hard = cur_hard
            update = True
        if cur_semihard <= best_semihard:
            best_semihard = cur_semihard
            update = True
        if update:
            model.save_weights(model_dir)
        else:
            patience_counter += 1

    # learning rate scheduler
    if patience_counter >= patience:
        alpha = alpha * decay
        tf.keras.backend.set_value(optimizer.lr, alpha)
        patience_counter = 0
        # early stopping
        if cur_all > 0.2 + best_all or \
            cur_hard > 0.2 + best_hard or \
                cur_semihard > 0.2 + best_semihard:
            break

logger.end('Done make_triplet.py')
