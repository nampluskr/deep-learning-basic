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
import tf_gan_trainer as gan


def get_parameters():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=2)
    p.add_argument("--noise_dim", type=int, default=100)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--keras", action='store_const', const=True, default=False)
    p.add_argument("--log_dir", type=str, default="tf_gan_mnist")
    return p.parse_args()


## MLP GAN/CGAN/ACGAN: Images - shape [N, 28*28] and values in [0, 1]
## MLP GAN/CGAN/ACGAN: Labels - shape [N, 10] in One-hot Encoding

@tf.function
def load_data(image, label):
    image = tf.reshape(image, [-1])
    image = tf.cast(image, dtype=tf.float32) / 255.

    label = tf.one_hot(label, depth=10)
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


def gan_mlp_generator():
    inputs = keras.Input(shape=(100,))
    x = layers.Dense(200)(inputs)
    x = layers.LeakyReLU(alpha=0.02)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(28*28)(x)
    outputs = layers.Activation(tf.sigmoid)(x)
    return keras.Model(inputs, outputs)


def gan_mlp_discriminator():
    inputs = keras.Input(shape=(28*28))
    x = layers.Flatten()(inputs)
    x = layers.Dense(200)(x)
    x = layers.LeakyReLU(alpha=0.02)(x)
    x = layers.LayerNormalization()(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


def make_noises(noise_size, noise_dim):
    noises = tf.random.normal((noise_size, noise_dim))
    return noises


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
    G = gan_mlp_generator()
    D = gan_mlp_discriminator()

    gan_mnist = gan.GAN(G, D, noise_dim=args.noise_dim)
    gan_mnist.compile(g_optim=keras.optimizers.Adam(learning_rate=1e-4),
                      d_optim=keras.optimizers.Adam(learning_rate=1e-4),
                      loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))

    ## Training:
    if args.keras:
        hist = gan_mnist.fit(train_loader, epochs=args.n_epochs)
    else:
        sample_noises = make_noises(noise_size=50, noise_dim=100)
        hist = gan.train(gan_mnist, train_loader, args, sample_noises)
        gan.plot_progress(hist, args)
