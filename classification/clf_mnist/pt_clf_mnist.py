import os
import torch
import torch.nn as nn

import numpy as np
import random

from dataset_mnist import load_mnist, load_fashion_mnist
import pt_clf_trainer as clf


def get_parameters():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--cnn", action='store_const', const=True, default=False)
    p.add_argument("--cpu", action='store_const', const=True, default=False)
    p.add_argument("--early_stop", action='store_const', const=True, default=False)
    p.add_argument("--min_loss", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_dir", type=str, default="pt_clf_mnist") # default: filename
    return p.parse_args()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.images, self.labels = data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image).float()/255.
        label = self.labels[idx]
        label = torch.tensor(label).long()
        return image, label


def get_dataloader(data, batch_size, training=True, use_cuda=False):
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    dataloader = torch.utils.data.DataLoader(dataset=Dataset(data),
                                             batch_size=batch_size,
                                             shuffle=training, **kwargs)
    return dataloader


def get_mlp_model(n_classes=10):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 200),
        nn.LeakyReLU(),
        nn.LayerNorm(200),
        nn.Linear(200, n_classes),)
    return model


class MlpModel(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 200),
            nn.LeakyReLU(),
            nn.LayerNorm(200),
            nn.Linear(200, n_classes),)

    def forward(self, x):
        return self.model(x)


def get_cnn_model(n_classes=10):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
        nn.Conv2d(1, 10, (5, 5), stride=2, padding=0), # [N,10,12,12]
        nn.LeakyReLU(),
        nn.BatchNorm2d(10),
        nn.Conv2d(10, 10, (3, 3), stride=2, padding=0), # [N,10,5,5]
        nn.LeakyReLU(),
        nn.BatchNorm2d(10),
        nn.Flatten(),
        nn.Linear(250, n_classes),)
    return model
  

if __name__ == "__main__":

    ## Parameters:
    args = get_parameters()

    manual_seed = 42
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.manual_seed(manual_seed)
    else:
        torch.manual_seed(manual_seed)

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
    train_loader = get_dataloader(train_data, args.batch_size, training=True, use_cuda=use_cuda)
    test_loader = get_dataloader(test_data, args.batch_size, training=False, use_cuda=use_cuda)

    ## Modeling:
    model = get_cnn_model().to(device) if args.cnn else get_mlp_model().to(device)
    clf_mnist = clf.Classifier(model)
    clf_mnist.compile(optim=torch.optim.Adam(model.parameters()),
                      loss_fn=nn.CrossEntropyLoss())

    ## Training:
    hist = clf.train(clf_mnist, train_loader, test_loader, args)
    clf.plot_progress(hist, args)