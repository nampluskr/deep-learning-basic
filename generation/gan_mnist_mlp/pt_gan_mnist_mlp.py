import os
import torch
import torch.nn as nn

import numpy as np
import random

from dataset_mnist import load_mnist, load_fashion_mnist
import pt_gan_trainer as gan


def get_parameters():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=2)
    p.add_argument("--noise_dim", type=int, default=100)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--cpu", action='store_const', const=True, default=False)
    p.add_argument("--log_dir", type=str, default="pt_gan_mnist") # default: filename
    return p.parse_args()


## MLP GAN/CGAN/ACGAN: Images - shape [N, 28*28] and values in [0, 1]
## MLP GAN/CGAN/ACGAN: Labels - shape [N, 10] in One-hot Encoding

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.images, self.labels = data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image).view(-1).float()/255.

        label = self.labels[idx]
        label = torch.tensor(label).long()
        label = nn.functional.one_hot(label, num_classes=10)
        return image, label


def get_dataloader(data, batch_size, training=True, use_cuda=False):
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    dataloader = torch.utils.data.DataLoader(dataset=Dataset(data),
                                             batch_size=batch_size,
                                             shuffle=training, **kwargs)
    return dataloader


def gan_mlp_generator():
    model = nn.Sequential(
            nn.Linear(100, 200), ## noise_dim
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid(),)
    return model


def gan_mlp_discriminator():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 200),
        nn.LeakyReLU(0.02),
        nn.LayerNorm(200),
        nn.Linear(200, 1),)
    return model


def make_noises(noise_size, noise_dim):
    noises = torch.randn(noise_size, noise_dim)
    return noises.to(device)


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
    G = gan_mlp_generator().to(device)
    D = gan_mlp_discriminator().to(device)

    gan_mnist = gan.GAN(G, D, noise_dim=args.noise_dim)
    gan_mnist.compile(g_optim=torch.optim.Adam(G.parameters(), lr=1e-4),
                      d_optim=torch.optim.Adam(D.parameters(), lr=1e-4),
                      loss_fn=nn.BCEWithLogitsLoss())

    ## Training:
    sample_noises = make_noises(noise_size=50, noise_dim=100)
    hist = gan.train(gan_mnist, train_loader, args, sample_noises)
    gan.plot_progress(hist, args)