import os
import torch
import torch.nn as nn
import torchvision

import numpy as np
import random

from dataset_cifar10 import load_cifar10
import pt_clf_trainer as clf


def get_parameters():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--cpu", action='store_const', const=True, default=False)
    p.add_argument("--early_stop", action='store_const', const=True, default=False)
    p.add_argument("--min_loss", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_dir", type=str, default="pt_clf_cifar10") # default: filename
    return p.parse_args()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.images, self.labels = data
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        label = torch.tensor(label).long()
        return image, label


def get_dataloader(data, batch_size, training=True, use_cuda=False):
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    
    if training:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    dataloader = torch.utils.data.DataLoader(dataset=Dataset(data, transform=transform),
                                             batch_size=batch_size,
                                             shuffle=training, **kwargs)
    return dataloader


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),)

    def forward(self, x):                 # (bs, in_channels,  h,  w)
        y = self.block(x)                 # (bs, out_channels, h/2, w/2)
        return y


class ClassifierModel(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(    # (bs, in_channels, w, h)
            ConvBlock(in_channels, 32),   # (bs,  32, w/2,  h/2)
            ConvBlock(32, 64),            # (bs,  64, w/4,  h/4)
            ConvBlock(64, 128),           # (bs, 128, w/8,  h/8)
            ConvBlock(128, 256),          # (bs, 256, w/16, h/16)
            ConvBlock(256, 512),)         # (bs, 512, w/32, h/32)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # (bs, 512, 1, 1)
            nn.Flatten(),                 # (bs, 512)
            nn.Linear(512, n_classes),    # (bs, n_classes)
            nn.LogSoftmax(dim=-1),)       # loss_fn = nn.NLLLoss()
            # nn.Softmax(dim=-1),)          # loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):                 # |x|: (bs, in_channels, w, h)
        z = self.features(x)              # |z|: (bs, 512, w/32, h/32)
        y = self.fc(z)                    # |y|: (bs, n_classes)
        return y                          # or F.log_softmax(y, dim=-1)


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
    data_path = '../../datasets/cifar10'
    train_data, test_data, class_names = load_cifar10(data_path, download=True)

    print(os.path.abspath(data_path))
    images, labels = train_data
    print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
    print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())
    print("classes:", class_names, '\n')

    ## Dataloaders:
    train_loader = get_dataloader(train_data, args.batch_size, training=True, use_cuda=use_cuda)
    test_loader = get_dataloader(test_data, args.batch_size, training=False, use_cuda=use_cuda)

    ## Modeling:
    model = ClassifierModel(in_channels=3, n_classes=10).to(device)
    clf_cifar10 = clf.Classifier(model)
    clf_cifar10.compile(optim=torch.optim.Adam(model.parameters()),
                        loss_fn=nn.CrossEntropyLoss())

    ## Training:
    hist = clf.train(clf_cifar10, train_loader, test_loader, args)
    clf.plot_progress(hist, args)