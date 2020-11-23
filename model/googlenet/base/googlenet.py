import numpy as np
import json
import tensorflow as tf

from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, Concatenate, Activation
from tensorflow.keras import Model
from tensorflow.nn import local_response_normalization
from tensorflow.keras.regularizers import l2

class LRN(Layer):
    def __init__(self, depth_radius=2, alpha=1.99999994948e-05, beta=0.75, name=None, trainable=True, dtype=tf.float32):
        super(LRN, self).__init__(name=name, trainable=trainable, dtype=dtype)
        self.depth_radius = depth_radius
        self.alpha = alpha
        self.beta = beta
    
    def call(self, input):
        return local_response_normalization(
            input, 
            depth_radius=self.depth_radius, 
            alpha=self.alpha, 
            beta=self.beta
        )

    def get_config(self):
        config = {
            'depth_radius': self.depth_radius,
            'alpha': self.alpha,
            'beta': self.beta
        }
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def Googlenet(weights_path=None, trainable=False):
    data = Input(shape=(224, 224, 3), name='data')

    conv1_7x7_s2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1', kernel_regularizer=l2(0.0002))(data)
    pool1_3x3_s2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1_7x7_s2)
    pool1_norm1 = LRN(depth_radius=2, alpha=1.99999994948e-05, beta=0.75, name='norm1')(pool1_3x3_s2)
    conv2_3x3_reduce = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2_1x1', kernel_regularizer=l2(0.0002))(pool1_norm1)
    conv2_3x3 = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
    conv2_norm2 = LRN(depth_radius=2, alpha=1.99999994948e-05, beta=0.75, name='norm2')(conv2_3x3)
    pool2_3x3_s2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2')(conv2_norm2)

    inception_3a_1x1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_3a_1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_reduce = Conv2D(96, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_3a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_3a_3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
    inception_3a_5x5_reduce = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_3a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_5x5 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', name='inception_3a_5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
    inception_3a_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3a_pool')(pool2_3x3_s2)
    inception_3a_pool_proj = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_3a_pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
    inception_3a_output = Concatenate(axis=3, name='inception_3a_output')([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj])

    inception_3b_1x1 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_3b_1x1', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_reduce = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_3b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3 = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_3b_3x3', kernel_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
    inception_3b_5x5_reduce = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_3b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_5x5 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', name='inception_3b_5x5', kernel_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
    inception_3b_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3b_pool')(inception_3a_output)
    inception_3b_pool_proj = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_3b_pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
    inception_3b_output = Concatenate(axis=3, name='inception_3b_output')([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj])
    pool3_3x3_s2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(inception_3b_output)

    inception_4a_1x1 = Conv2D(192, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4a_1x1', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_reduce = Conv2D(96, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3 = Conv2D(208, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_4a_3x3' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
    inception_4a_5x5_reduce = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_5x5 = Conv2D(48, (5, 5), strides=(1, 1), padding='same', activation='relu', name='inception_4a_5x5', kernel_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
    inception_4a_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4a_pool')(pool3_3x3_s2)
    inception_4a_pool_proj = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4a_pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
    inception_4a_output = Concatenate(axis=3, name='inception_4a_output')([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj])

    inception_4b_1x1 = Conv2D(160, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4b_1x1', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_reduce = Conv2D(112, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3 = Conv2D(224, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_4b_3x3', kernel_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
    inception_4b_5x5_reduce = Conv2D(24, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_5x5 = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', name='inception_4b_5x5', kernel_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
    inception_4b_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4b_pool')(inception_4a_output)
    inception_4b_pool_proj = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4b_pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)
    inception_4b_output = Concatenate(axis=3, name='inception_4b_output')([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj])

    inception_4c_1x1 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4c_1x1', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_reduce = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4c_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_4c_3x3', kernel_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
    inception_4c_5x5_reduce = Conv2D(24, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4c_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_5x5 = Conv2D(64, (5 ,5), strides=(1, 1), padding='same', activation='relu', name='inception_4c_5x5', kernel_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
    inception_4c_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4c_pool')(inception_4b_output)
    inception_4c_pool_proj = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4c_pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)
    inception_4c_output = Concatenate(axis=3, name='inception_4c_output')([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj])

    inception_4d_1x1 = Conv2D(112, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4d_1x1', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_reduce = Conv2D(144, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4d_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3 = Conv2D(288, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_4d_3x3', kernel_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
    inception_4d_5x5_reduce = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4d_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_5x5 = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', name='inception_4d_5x5', kernel_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
    inception_4d_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4d_pool')(inception_4c_output)
    inception_4d_pool_proj = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4d_pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)
    inception_4d_output = Concatenate(axis=3, name='inception_4d_output')([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj])

    inception_4e_1x1 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4e_1x1', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_reduce = Conv2D(160, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4e_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3 = Conv2D(320, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_4e_3x3', kernel_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
    inception_4e_5x5_reduce = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4e_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_5x5 = Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', name='inception_4e_5x5', kernel_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
    inception_4e_pool = MaxPool2D(pool_size=(3, 3), strides=(1,1), padding='same', name='inception_4e_pool')(inception_4d_output)
    inception_4e_pool_proj = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_4e_pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)
    inception_4e_output = Concatenate(axis=3, name='inception_4e_output')([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj])
    pool4_3x3_s2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool4')(inception_4e_output)

    inception_5a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_5a_1x1', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3_reduce = Conv2D(160, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_5a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3 = Conv2D(320, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_5a_3x3', kernel_regularizer=l2(0.0002))(inception_5a_3x3_reduce)
    inception_5a_5x5_reduce = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_5a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_5x5 = Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', name='inception_5a_5x5', kernel_regularizer=l2(0.0002))(inception_5a_5x5_reduce)
    inception_5a_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5a_pool')(pool4_3x3_s2)
    inception_5a_pool_proj = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_5a_pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)
    inception_5a_output = Concatenate(axis=3, name='inception_5a_output')([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj])

    inception_5b_1x1 = Conv2D(384, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_5b_1x1', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3_reduce = Conv2D(192, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_5b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3 = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', name='inception_5b_3x3', kernel_regularizer=l2(0.0002))(inception_5b_3x3_reduce)
    inception_5b_5x5_reduce = Conv2D(48, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_5b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_5x5 = Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', name='inception_5b_5x5', kernel_regularizer=l2(0.0002))(inception_5b_5x5_reduce)
    inception_5b_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5b_pool')(inception_5a_output)
    inception_5b_pool_proj = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', name='inception_5b_pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)
    inception_5b_output = Concatenate(axis=3, name='inception_5b_output')([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj])

    # pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='pool5')(inception_5b_output)
    pool5_7x7_s1 = GlobalAveragePooling2D(name='pool5')(inception_5b_output)
    # loss3_classifier = Dense(431, name='loss3_classifier_model', kernel_regularizer=l2(0.0002))(pool5_7x7_s1)
    # loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

    googlenet = Model(inputs=data, outputs=pool5_7x7_s1, name='GoogleNet')

    if weights_path is not None:
        if weights_path.endswith('.npy'):
            pretrained_weights = np.load(weights_path, encoding='bytes', allow_pickle=True).item()
            for layer in googlenet.layers:
                loaded_weight = list()
                for weight in layer.trainable_weights:
                    name = weight.name[:-2].replace('kernel', 'weights').replace('bias', 'biases').split('/')
                    value = pretrained_weights[name[0]][name[1].encode()]
                    weight.assign(value)
        elif weights_path.endswith('.h5'):
            googlenet.load_weights(weights_path)
        else:
            raise Exception('Only .npy or .h5 weights are supported')
        googlenet.trainable = trainable

    else:
        googlenet.trainable = True
    
    return googlenet