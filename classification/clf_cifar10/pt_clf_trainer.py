import os
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import sys
from tqdm import tqdm
from copy import deepcopy
import pathlib


class Classifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def metric_fn(self, y_pred, y):
        return torch.eq(y_pred.argmax(-1), y).float().mean()
    
    def compile(self, optim, loss_fn):
        self.optim = optim
        self.loss_fn = loss_fn

    def train_step(self, data):
        x, y = data
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        acc = self.metric_fn(y_pred, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {'loss':loss, 'acc':acc}

    @torch.no_grad()
    def test_step(self, data):
        x, y = data
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        acc = self.metric_fn(y_pred, y)
        return {'loss':loss, 'acc':acc}


def train(clf, train_loader, valid_loader, args):
    log_path = os.path.join(os.getcwd(), args.log_dir)
    if not pathlib.Path(log_path).exists():
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_path, args.log_dir + "_log.txt")
    print_parameters(args, log_file)
        
    device = next(clf.parameters()).device
    hist = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
    best_loss, counter = 1e12, 1
    for epoch in range(args.n_epochs):

        ## Training
        desc = "Epoch[%3d/%3d]" % (epoch+1, args.n_epochs)
        with tqdm(train_loader, total=len(train_loader), ncols=100,
                file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)

            loss, acc = np.asfarray([]), np.asfarray([])
            for x, y in pbar:
                data = x.to(device), y.to(device)
                results = clf.train_step(data)
                loss = np.append(loss, results['loss'].item())
                acc = np.append(acc, results['acc'].item())
                pbar.set_postfix({'loss': "%.4f" % loss.mean(), 'acc': "%.4f" % acc.mean()})

            hist['loss'].append(loss.mean())
            hist['acc'].append(acc.mean())

        ## Validation
        clf.model.eval()
        val_loss, val_acc = np.asfarray([]), np.asfarray([])
        for x, y in valid_loader:
            data = x.to(device), y.to(device)
            results = clf.test_step(data)
            val_loss = np.append(val_loss, results['loss'].item())
            val_acc = np.append(val_acc, results['acc'].item())

        hist['val_loss'].append(val_loss.mean())
        hist['val_acc'].append(val_acc.mean())
        
        if val_loss.mean() < best_loss and (best_loss - val_loss.mean()) > args.min_loss:
            best_loss = val_loss.mean()
            best_epoch = epoch + 1
            best_model = deepcopy(clf.model.state_dict())
            counter = 1
        else:
            counter += 1

        ## Print log
        if (epoch + 1) % args.log_interval == 0 or args.early_stop:
            desc += ": loss=%.4f, acc=%.4f" % (loss.mean(), acc.mean())
            desc += " - val_loss=%.4f, val_acc=%.4f (%d)" % (val_loss.mean(), val_acc.mean(), counter)
            print_log(desc, log_file)

        ## Early stopping            
        if args.early_stop and counter == args.patience:
            print_log("Early stopped! (Best epoch=%d)" % best_epoch, log_file)
            break

    clf.model.load_state_dict(best_model)
    torch.save(best_model, os.path.join(log_path, args.log_dir + "_weights.pth"))
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