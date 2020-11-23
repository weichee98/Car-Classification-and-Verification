import os
import argparse

from utils.logger import Logger
from utils.dict_json import write_dict_to_json
from src.input_pipeline import InputPipeline
from model.model_manager import load_model, get_input_size

import pandas as pd
import numpy as np
import tensorflow as tf


logger = Logger()
logger.log('Start model_test.py')


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='Path to test dataset csv file', required=True)
parser.add_argument('-i', '--input_dir', help='Path to the image directory', required=True)
parser.add_argument('-mc', '--model_config', help='Path to the saved model architecture', required=True)
parser.add_argument('-mw', '--model_weights', help='Path to the saved model weights', default=None)
parser.add_argument('-b', '--batch_size', type=int, default=500)
args = parser.parse_args()


# Load dataset
logger.log('Load dataset')
df = pd.read_csv(args.data)
input_dir = args.input_dir if args.input_dir[-1] == '/' else args.input_dir + '/'
df['filename'] = input_dir + \
                 df['make_id'].astype(str) + '/' + \
                 df['model_id'].astype(str) + '/' + \
                 df['released_year'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'unknown') + '/' + \
                 df['image_name']


# Load model
logger.log('Load model')
model = load_model(args.model_config, args.model_weights)


# Prepare dataset
logger.log('Prepare dataset')
img_size = get_input_size(model.name)
batch_size = args.batch_size
pipeline = InputPipeline(img_size=img_size)
test_ds = pipeline.load_dataset(df['filename'], df['model_id'], batch_size=batch_size, mode='test')
TEST_STEPS = len(df) // batch_size


# Test
logger.log('Test model')

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
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
model.trainable = False
results = model.evaluate(test_ds, batch_size=batch_size, verbose=0, return_dict=True)
print(results)


# Save results
if not os.path.exists('./log'):
    os.mkdir('./log')
if not os.path.exists(f'./log/{model.name}'):
    os.mkdir(f'./log/{model.name}')
log_dir = f'./log/{model.name}/' + args.model_weights.split('/')[-1] + '_test.log'

write_dict_to_json(results, log_dir)
logger.log('Saved result to \"' + log_dir + '\"')

logger.end('Done model_test.py')