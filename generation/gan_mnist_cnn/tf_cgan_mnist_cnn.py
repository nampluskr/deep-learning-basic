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
    p.add_argument("--log_dir", type=str, default="tf_cgan_mnist")
    return p.parse_args()


## CNN GAN/CGAN/ACGAN: Images - shape [N, 28, 28, 1] and values in [-1, 1]
## CNN GAN/CGAN/ACGAN: Labels - shape [N]

@tf.function
def load_data(image, label):
    image = tf.expand_dims(image, axis=-1)
    image = tf.cast(image, dtype=tf.float32) / 255.
    image = (image - 0.5)/0.5

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


def cgan_cnn_generator(embedding_dim=100):
    noises_reshape = keras.models.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(7*7*256, use_bias=False),
        layers.ReLU(),
        layers.Reshape((7, 7, 256)),])

    labels_reshape = keras.models.Sequential([
        layers.Input(shape=(1,)),
        layers.Embedding(10, embedding_dim),
        layers.Dense(7*7*1),
        layers.Reshape((7, 7, 1)),])

    model = keras.models.Sequential([
        layers.Input(shape=(7, 7, 256 + 1)),
        layers.Conv2DTranspose(128, (5, 5), strides=1, padding='same',
                                        use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same',
                                        use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=2, padding='same',
                                        use_bias=False),
        layers.Activation(tf.tanh),])

    noises = keras.Input(shape=(100,))
    labels = keras.Input(shape=(1,))

    noises_labels = layers.Concatenate()([noises_reshape(noises), labels_reshape(labels)])
    outputs = model(noises_labels)
    return keras.Model(inputs=[noises, labels], outputs=outputs)


def cgan_cnn_discriminator(embedding_dim=100):
    labels_reshape = keras.models.Sequential([
        layers.Input(shape=(1,)),
        layers.Embedding(10, embedding_dim),
        layers.Dense(28*28*1),
        layers.Reshape((28, 28, 1)),])

    model = keras.models.Sequential([
        layers.Input(shape=(28, 28, 1 + 1)),
        layers.Conv2D(64, (5, 5), strides=2, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, (5, 5), strides=2, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(256, (5, 5), strides=1, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1),])

    images = layers.Input(shape=(28, 28, 1))
    labels = layers.Input(shape=(1,))

    images_labels = layers.Concatenate()([images, labels_reshape(labels)])
    outputs = model(images_labels)
    return keras.Model(inputs=[images, labels], outputs=outputs)


def make_noises_labels(noise_size, noise_dim):
    noises = tf.random.normal((noise_size, noise_dim))
    labels = tf.transpose(tf.reshape(tf.repeat(tf.range(10), 5), (-1, 5)))
    labels = tf.cast(tf.reshape(labels, -1), tf.int64)
    return noises, labels


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
    G = cgan_cnn_generator()
    D = cgan_cnn_discriminator()

    cgan_mnist = gan.CondGAN(G, D, noise_dim=args.noise_dim)
    cgan_mnist.compile(g_optim=keras.optimizers.Adam(learning_rate=1e-4),
                d_optim=keras.optimizers.Adam(learning_rate=1e-4),
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))

    ## Training:
    if args.keras:
        hist = cgan_mnist.fit(train_loader, epochs=args.n_epochs)
    else:
        sample_noises = make_noises_labels(noise_size=50, noise_dim=100)
        hist = gan.train(cgan_mnist, train_loader, args, sample_noises)
        gan.plot_progress(hist, args)
