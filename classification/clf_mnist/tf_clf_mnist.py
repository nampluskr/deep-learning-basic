import os
from tensorflow.python.keras.utils.generic_utils import default
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## Use CPU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random

from dataset_mnist import load_mnist, load_fashion_mnist
import tf_clf_trainer as clf


def get_parameters():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--cnn", action='store_const', const=True, default=False)
    p.add_argument("--keras", action='store_const', const=True, default=False)
    p.add_argument("--early_stop", action='store_const', const=True, default=False)
    p.add_argument("--min_loss", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_dir", type=str, default="tf_clf_mnist") # default: filename
    return p.parse_args()


@tf.function
def load_data(image, label):
    image = tf.cast(image, dtype=tf.float32)/255.
    label = tf.cast(label, dtype=tf.int64)
    return image, label


def get_dataloader(dataset, batch_size, training=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataloader = tf.data.Dataset.from_tensor_slices(dataset)
    dataloader = dataloader.map(load_data, num_parallel_calls=AUTOTUNE)
    if training:
        dataloader = dataloader.shuffle(10000)
    dataloader = dataloader.batch(batch_size).prefetch(AUTOTUNE)
    return dataloader


def get_mlp_model(n_classes=10):
    inputs = keras.Input(shape=(28, 28))
    x = layers.Flatten()(inputs)
    x = layers.Dense(200)(x)
    x = layers.LeakyReLU()(x)
    x = layers.LayerNormalization()(x)
    outputs = layers.Dense(n_classes)(x)
    return keras.Model(inputs, outputs)


def get_cnn_model(n_classes=10):
    inputs = keras.Input(shape=(28, 28))
    x = layers.Reshape((28, 28, 1))(inputs)
    x = layers.Conv2D(10, (5, 5), strides=2, padding='valid')(x) # [N,12,12,10]
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(10, (3, 3), strides=2, padding='valid')(x) # [N,5,5,10]
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(n_classes)(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":

    ## Parameters:
    args = get_parameters()

    manual_seed = 42
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    tf.random.set_seed(manual_seed)

    ## Dataset:
    if args.fashion:
        data_path = '../../datasets/fashion_mnist'
        train_data, test_data, class_names = load_fashion_mnist(data_path, download=True)
    else:
        data_path = '../../datasets/mnist'
        train_data, test_data, class_names = load_mnist(data_path, download=True)

    print(os.path.abspath(data_path))
    images, labels = train_data
    print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
    print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())
    print("classes:", class_names, '\n')

    ## Dataloaders:
    train_loader = get_dataloader(train_data, args.batch_size, training=True)
    test_loader = get_dataloader(test_data, args.batch_size, training=False)

    ## Modeling:
    model = get_cnn_model() if args.cnn else get_mlp_model()
    clf_mnist = clf.Classifier(model)
    clf_mnist.compile(optim=keras.optimizers.Adam(),
                loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    ## Training:
    if args.keras:
        hist = clf_mnist.fit(train_loader, validation_data=test_loader, epochs=args.n_epochs)
    else:
        hist = clf.train(clf_mnist, train_loader, test_loader, args)
        clf.plot_progress(hist, args)