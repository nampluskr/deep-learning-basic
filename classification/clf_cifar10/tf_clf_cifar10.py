import os
from tensorflow.python.keras.utils.generic_utils import default
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## Use CPU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

import random
import sys
from tqdm import tqdm
from copy import deepcopy
import pathlib

from dataset_cifar10 import load_cifar10
import tf_clf_trainer as clf


def get_parameters():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--keras", action='store_const', const=True, default=False)
    p.add_argument("--early_stop", action='store_const', const=True, default=False)
    p.add_argument("--min_loss", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_dir", type=str, default="tf_clf_cifar10") # default: filename
    return p.parse_args()


@tf.function
def augmentation(image, label):
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_crop(image, size=input_shape)
    image = tf.image.random_brightness(image, max_delta=0.5)
    return image, label


@tf.function
def load_data(image, label):
    image = tf.cast(image, dtype=tf.float32)/255.
    image -= tf.constant([0.4914, 0.4822, 0.4465], shape=[1, 1, 3]) # mean
    image /= tf.constant([0.2023, 0.1994, 0.2010], shape=[1, 1, 3]) # std
    label = tf.cast(label, dtype=tf.int64)
    return image, label


def get_dataloader(dataset, batch_size, training=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataloader = tf.data.Dataset.from_tensor_slices(dataset)
    if training:
        dataloader = dataloader.map(augmentation, num_parallel_calls=AUTOTUNE)
        dataloader = dataloader.shuffle(10000)
    dataloader = dataloader.map(load_data, num_parallel_calls=AUTOTUNE)
    dataloader = dataloader.batch(batch_size).prefetch(AUTOTUNE)
    return dataloader


class ConvBlock(keras.Model):
    def __init__(self, out_channels):
        super().__init__()
        self.block = keras.models.Sequential([
            layers.Conv2D(out_channels, (3, 3), strides=1, padding='same'),
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(out_channels, (3, 3), strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.BatchNormalization(),])

    def call(self, x, training=False):             # (bs, h, w, in_channels)
        x = self.block(x, training=training)       # (bs, h/2, w/2, out_channels)
        return x


class ClassifierModel(keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.features = keras.models.Sequential([  # (bs, w, h, in_channels)
            # keras.layers.Input(shape=input_shape),
            ConvBlock(32),                         # (bs, w/2,  h/2,   32)
            ConvBlock(64),                         # (bs, w/4,  h/4,   64)
            ConvBlock(128),                        # (bs, w/8,  h/8,  128)
            ConvBlock(256),                        # (bs, w/16, h/16, 256)
            ConvBlock(512),])                      # (bs, w/32, h/32, 512)

        self.fc = keras.models.Sequential([
            layers.GlobalAveragePooling2D(), # (bs, 512)
            layers.Dense(n_classes),])

    def call(self, x, training=False):             # |x|: (bs, w, h, in_channels)
        z = self.features(x, training=training)    # |z|: (bs, w/32, h/32, 512)
        y = self.fc(z, training=training)          # |y|: (bs, n_classes)
        return y


if __name__ == "__main__":

    ## Parameters:
    args = get_parameters()

    manual_seed = 42
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    tf.random.set_seed(manual_seed)

    ## Dataset:
    data_path = '../../datasets/cifar10'
    train_data, test_data, class_names = load_cifar10(data_path, download=True)

    print(os.path.abspath(data_path))
    images, labels = train_data
    print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
    print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())
    print("classes:", class_names, '\n')

    # ## Dataloaders:
    train_loader = get_dataloader(train_data, args.batch_size, training=True)
    test_loader = get_dataloader(test_data, args.batch_size, training=False)

    ## Modeling:
    model = ClassifierModel(n_classes=10)
    clf_mnist = clf.Classifier(model)
    clf_mnist.compile(optim=keras.optimizers.Adam(),
                      loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # ## Training:
    if args.keras:
        hist = clf_mnist.fit(train_loader, validation_data=test_loader, epochs=args.n_epochs)
    else:
        hist = clf.train(clf_mnist, train_loader, test_loader, args)
        clf.plot_progress(hist, args)