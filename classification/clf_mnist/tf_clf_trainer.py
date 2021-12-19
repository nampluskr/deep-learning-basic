import os
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from copy import deepcopy
import pathlib


class Classifier(keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.batch_loss = keras.metrics.Mean()
        self.batch_acc = keras.metrics.Mean()

    @tf.function
    def metric_fn(self, y, y_pred):
        return tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(y_pred, -1), y), tf.float32))

    def compile(self, optim, loss_fn):
        super().compile()
        self.optim = optim
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.batch_loss, self.batch_acc]

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))

        self.batch_loss.update_state(loss)
        self.batch_acc.update_state(self.metric_fn(y, y_pred))
        return {"loss": self.batch_loss.result(), "acc": self.batch_acc.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        self.batch_loss.update_state(self.loss_fn(y, y_pred))
        self.batch_acc.update_state(self.metric_fn(y, y_pred))
        return {"loss": self.batch_loss.result(), "acc": self.batch_acc.result()}


def train(clf, train_loader, valid_loader, args):
    log_path = os.path.join(os.getcwd(), args.log_dir)
    if not pathlib.Path(log_path).exists():
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_path, args.log_dir + "_log.txt")
    print_parameters(args, log_file)

    hist = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
    n_batches = tf.data.experimental.cardinality(train_loader).numpy()
    best_loss, counter = 1e12, 1
    for epoch in range(args.n_epochs):

        ## Training
        desc = "Epoch[%3d/%3d]" % (epoch+1, args.n_epochs)
        with tqdm(train_loader, total=n_batches, ncols=100,
                file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)

            clf.batch_loss.reset_states()
            clf.batch_acc.reset_states()
            for data in pbar:
                clf.train_step(data)
                loss = clf.batch_loss.result().numpy()
                acc = clf.batch_acc.result().numpy()
                pbar.set_postfix({'loss': "%.4f" % loss, 'acc': "%.4f" % acc})

            hist['loss'].append(loss)
            hist['acc'].append(acc)

        ## Validation
        clf.batch_loss.reset_states()
        clf.batch_acc.reset_states()
        for val_data in valid_loader:
            clf.test_step(val_data)
        val_loss = clf.batch_loss.result().numpy()
        val_acc = clf.batch_acc.result().numpy()

        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)

        if val_loss < best_loss and (best_loss - val_loss) > args.min_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_model = deepcopy(clf.model.get_weights())
            counter = 1
        else:
            counter += 1

        ## Print log
        if (epoch + 1) % args.log_interval == 0 or args.early_stop:
            desc += ": loss=%.4f, acc=%.4f" % (loss, acc)
            desc += " - val_loss=%.4f, val_acc=%.4f (%d)" % (val_loss, val_acc, counter)
            print_log(desc, log_file)

        ## Early stopping
        if args.early_stop and counter == args.patience:
            print_log("Early stopped! (Best epoch=%d)" % best_epoch, log_file)
            break

    clf.model.set_weights(best_model)
    clf.model.save_weights(os.path.join(log_path, args.log_dir + "_weights.h5"))
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