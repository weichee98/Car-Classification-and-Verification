import numpy as np
import json
import tensorflow as tf

# from model.googlenet.base.googlenet import Googlenet, LRN


def save_model(model, model_config, model_weights=None):
    config = model.to_json()
    with open(model_config, 'w') as file:
        json_string = json.dumps(config, indent=4)
        file.write(json_string)
    if model_weights:
        model.save_weights(model_weights)


def load_model(model_config, model_weights=None):
    with open(model_config) as file:
        config = json.load(file)
    try:
        model = tf.keras.models.model_from_json(config)
    except ValueError:
        model = tf.keras.models.model_from_json(config, custom_objects={'LRN': LRN})
    if model_weights:
        model.load_weights(model_weights)
    return model


def get_input_size(model_name):
    input_sizes = {
        'inception_v3': (299, 299),
        'googlenet': (224, 224)
    }
    for k in input_sizes.keys():
        if k in model_name:
            return input_sizes[k]
    else:
        raise Exception('Cannot get input size of model, please define it explicitly')


if __name__ == '__main__':
    pass