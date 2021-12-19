import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import sys
from tqdm import tqdm
import pathlib


class GAN(keras.Model):
    def __init__(self, generator, discriminator, noise_dim):
        super().__init__()
        self.G, self.D = generator, discriminator
        self.noise_dim = noise_dim
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")

    def compile(self, g_optim, d_optim, loss_fn):
        super().compile()
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    @tf.function
    def train_step(self, data):
        r_images, _ = data

        ## Train the discriminator:
        noises = tf.random.normal((tf.shape(r_images)[0], self.noise_dim))
        f_images = self.G(noises, training=False)
        with tf.GradientTape() as d_tape:
            r_logits = self.D(r_images, training=True)
            f_logits = self.D(f_images, training=True)

            r_loss = self.loss_fn(tf.ones_like(r_logits), r_logits)
            f_loss = self.loss_fn(tf.zeros_like(f_logits), f_logits)
            d_loss = r_loss + f_loss

        d_grads = d_tape.gradient(d_loss, self.D.trainable_variables)
        self.d_optim.apply_gradients(zip(d_grads, self.D.trainable_variables))

        ## Train the generator:
        noises = tf.random.normal((tf.shape(r_images)[0], self.noise_dim))
        with tf.GradientTape() as g_tape:
            g_images = self.G(noises, training=True)
            g_logits = self.D(g_images, training=False)
            g_loss = self.loss_fn(tf.ones_like(g_logits), g_logits)

        g_grads = g_tape.gradient(g_loss, self.G.trainable_variables)
        self.g_optim.apply_gradients(zip(g_grads, self.G.trainable_variables))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {"d_loss": self.d_loss_tracker.result(),
                "g_loss": self.g_loss_tracker.result()}


class CondGAN(GAN):
    def __init__(self, generator, discriminator, noise_dim):
        super().__init__(generator, discriminator, noise_dim)

    @tf.function
    def train_step(self, data):
        r_images, labels = data
        
        ## Train the discriminator:
        noises = tf.random.normal((tf.shape(r_images)[0], self.noise_dim))
        f_images = self.G([noises, labels], training=False)
        with tf.GradientTape() as d_tape:
            r_logits = self.D([r_images, labels], training=True)
            f_logits = self.D([f_images, labels], training=True)

            r_loss = self.loss_fn(tf.ones_like(r_logits), r_logits)
            f_loss = self.loss_fn(tf.zeros_like(f_logits), f_logits)
            d_loss = r_loss + f_loss

        d_grads = d_tape.gradient(d_loss, self.D.trainable_variables)
        self.d_optim.apply_gradients(zip(d_grads, self.D.trainable_variables))

        ## Train the generator:
        noises = tf.random.normal((tf.shape(r_images)[0], self.noise_dim))
        with tf.GradientTape() as g_tape:
            g_images = self.G([noises, labels], training=True)
            g_logits = self.D([g_images, labels], training=False)
            g_loss = self.loss_fn(tf.ones_like(g_logits), g_logits)

        g_grads = g_tape.gradient(g_loss, self.G.trainable_variables)
        self.g_optim.apply_gradients(zip(g_grads, self.G.trainable_variables))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {"d_loss": self.d_loss_tracker.result(), 
                "g_loss": self.g_loss_tracker.result()}


class AuxCondGAN(GAN):
    def __init__(self, generator, discriminator, noise_dim):
        super().__init__(generator, discriminator, noise_dim)

    def compile(self, g_optim, d_optim, loss_fn, loss_fn_aux):
        super().compile(g_optim, d_optim, loss_fn)
        self.loss_fn_aux = loss_fn_aux

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    @tf.function
    def train_step(self, data):
        r_images, labels = data
        
        ## Train the discriminator:
        noises = tf.random.normal((tf.shape(r_images)[0], self.noise_dim))
        f_images = self.G([noises, labels], training=False)
        with tf.GradientTape() as d_tape:
            r_logits, r_preds = self.D([r_images, labels], training=True)
            f_logits, f_preds = self.D([f_images, labels], training=True)

            r_loss = self.loss_fn(tf.ones_like(r_logits), r_logits)
            f_loss = self.loss_fn(tf.zeros_like(f_logits), f_logits)

            d_loss = r_loss + f_loss
            d_loss += self.loss_fn_aux(labels, r_preds)
            d_loss += self.loss_fn_aux(labels, f_preds)

        d_grads = d_tape.gradient(d_loss, self.D.trainable_variables)
        self.d_optim.apply_gradients(zip(d_grads, self.D.trainable_variables))

        ## Train the generator:
        noises = tf.random.normal((tf.shape(r_images)[0], self.noise_dim))
        with tf.GradientTape() as g_tape:
            g_images = self.G([noises, labels], training=True)
            g_logits, g_preds = self.D([g_images, labels], training=False)
            g_loss = self.loss_fn(tf.ones_like(g_logits), g_logits)
            g_loss += self.loss_fn_aux(labels, g_preds)

        g_grads = g_tape.gradient(g_loss, self.G.trainable_variables)
        self.g_optim.apply_gradients(zip(g_grads, self.G.trainable_variables))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {"d_loss": self.d_loss_tracker.result(), 
                "g_loss": self.g_loss_tracker.result()}
        
        
def train(gan, train_loader, args, sample_noises):
    log_path = os.path.join(os.getcwd(), args.log_dir)
    if not pathlib.Path(log_path).exists():
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_path, args.log_dir + "_log.txt")
    print_parameters(args, log_file)
    
    hist = {'d_loss':[], 'g_loss':[]}
    n_batches = tf.data.experimental.cardinality(train_loader).numpy()

    for epoch in range(args.n_epochs):
        desc = "Epoch[%3d/%3d]" % (epoch+1, args.n_epochs)
        with tqdm(train_loader, total=n_batches, ncols=100,
                file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)

            gan.d_loss_tracker.reset_states()
            gan.g_loss_tracker.reset_states()

            for data in pbar:
                gan.train_step(data)

                d_loss = gan.d_loss_tracker.result().numpy()
                g_loss = gan.g_loss_tracker.result().numpy()
                pbar.set_postfix(d_loss="%.3f" % d_loss, g_loss="%.3f" % g_loss)

            hist['d_loss'].append(d_loss)
            hist['g_loss'].append(g_loss)

        print_log(desc + ": d_loss=%.3f, g_loss=%.3f" % (d_loss, g_loss), log_file)

        if (epoch + 1) % args.log_interval == 0:
            sample_images = gan.G.predict(sample_noises).reshape(-1, 28, 28)
            sample_images = sample_images*255
            img_name = os.path.join(log_path, args.log_dir + "-%04d.png" % (epoch+1))
            save_images(sample_images, img_name=img_name)
            
    gan.G.save_weights(os.path.join(log_path, args.log_dir + "_weights.h5"))
    return hist


def print_parameters(args, log_file):
    parameters = ""
    for key, value in vars(args).items():
        parameters += "%s=%s, " % (key, str(value))
    print(parameters[:-2] + '\n')

    with open(log_file, 'w') as f:
        f.write(parameters[:-2] + '\n\n')


def print_log(desc, log_file):
    print(desc)
    with open(log_file, 'a') as f:
        f.write(desc + '\n')
        

def save_images(images, labels=None, n_cols=10, width=8, img_name=""):
    n_rows = images.shape[0] // n_cols + (1 if images.shape[0] % n_cols else 0)
    height = width*n_rows/n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))
    for i, ax in enumerate(axes.flat):
        if i < images.shape[0]:
            ax.imshow(images[i].astype('uint8'), interpolation='none', cmap='gray_r')
            if labels is not None:
                ax.set_title(labels[i])
        ax.set_axis_off()
    fig.tight_layout()

    if labels is None:
        plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(img_name, pad_inches=0)
    plt.close()


def plot_progress(hist, args, skip=1):
    fig, ax = plt.subplots(figsize=(8,4))
    for name, loss in hist.items():
        iter = range(1, len(loss) + 1)
        ax.plot(iter[::skip], loss[::skip], 'o-', label=name)
    ax.set_title(args.log_dir, fontsize=15)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(color='k', ls=':', lw=1)
    fig.tight_layout()

    img_name = os.path.join(os.getcwd(), args.log_dir, args.log_dir + "_hist.png")
    plt.savefig(img_name, pad_inches=0)
    plt.close()


if __name__ == "__main__":

    pass