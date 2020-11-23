import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

class InputPipeline:

    def __init__(self, img_size=(64, 64)):
        self.__IMG_SIZE = img_size

    def __parse_function(self, filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image_string, channels=3)
        return image, label

    def __image_rescale_resize(self, image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, self.__IMG_SIZE)
        return image, label

    def __image_augmentation(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(
            image, 
            size=(
                np.random.randint(0.8 * self.__IMG_SIZE[0], self.__IMG_SIZE[0]), 
                np.random.randint(0.8 * self.__IMG_SIZE[1], self.__IMG_SIZE[1]), 
                3
            )
        )
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_contrast(image, 0.7, 1)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_saturation(image, 0.5, 1.5)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.resize(image, self.__IMG_SIZE)
        return image, label

    def load_dataset(self, filenames, labels, batch_size=30, mode='train'):
        """
        :param filenames: image paths
        :param labels: image labels
        :param batch_size: number of images per batch
        :param mode: 'train' or 'eval'
        :return: tf.data.Dataset
        """
        filenames, labels = shuffle(filenames, labels)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels - 1))
        dataset = dataset.map(self.__parse_function, num_parallel_calls=4)
        dataset = dataset.map(self.__image_rescale_resize, num_parallel_calls=4)
        if mode == 'train':
            dataset = dataset.map(self.__image_augmentation, num_parallel_calls=4)
            dataset = dataset.shuffle(batch_size).repeat()
        elif mode == 'valid':
            dataset = dataset.shuffle(batch_size).repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def preprocess_image(self, image, label=None, augment=False):
        image, label = self.__image_rescale_resize(image, label)
        if augment:
            image, label = self.__image_augmentation(image, label)
        return image, label

    def load_image(self, filename, label=None, augment=False):
        image, label = self.__parse_function(filename, label)
        return self.preprocess_image(image, label, augment)

    def load_balanced_dataset(self, filenames, labels, batch_size=30, mode='train'):
        unique_labels = np.unique(labels)
        all_ds = list()
        for l in unique_labels:
            idx = np.flatnonzero(labels == l)
            label = labels[idx]
            filename = filenames[idx]
            ds = tf.data.Dataset.from_tensor_slices((filename, label - 1)).shuffle(5).repeat()
            all_ds.append(ds)
        # choice_ds = tf.data.Dataset.range(len(all_ds)).repeat()
        # dataset = tf.data.experimental.choose_from_datasets(all_ds, choice_ds)
        dataset = tf.data.experimental.sample_from_datasets(all_ds)
        dataset = dataset.map(self.__parse_function, num_parallel_calls=4)
        dataset = dataset.map(self.__image_rescale_resize, num_parallel_calls=4)
        if mode == 'train':
            dataset = dataset.map(self.__image_augmentation, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset