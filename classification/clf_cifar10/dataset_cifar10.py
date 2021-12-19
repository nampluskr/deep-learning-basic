import os
import pathlib
import numpy as np
import pickle


def download_data(url, data_path):
    import requests
    from tqdm import tqdm

    files_size = int(requests.head(url).headers["content-length"])
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(data_path, os.path.basename(url))

    pbar = tqdm(total=files_size, unit='B', unit_scale=True, unit_divisor=1024,
                ascii=True, desc=os.path.basename(url), ncols=100)

    with requests.get(url, stream=True) as req, open(file_path, 'wb') as file:
        for chunk in req.iter_content(chunk_size=1024):
            data_size = file.write(chunk)
            pbar.update(data_size)
        pbar.close()


def extract(file_path, extract_path):
    import shutil

    print(">> Extracting", os.path.basename(file_path))
    shutil.unpack_archive(file_path, extract_path)
    pathlib.Path(file_path).unlink()
    print(">> Complete!")


def load(file_path, image=False):
    import gzip
    import numpy as np

    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if image else 8)
    return data.reshape(-1, 28, 28) if image else data


def unpickle(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f, encoding='bytes')

    x = np.array(file[b'data'])
    y = np.array(file[b'labels'])
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), y


def load_cifar10(data_path, download=False):
    """ The MNIST dataset: http://yann.lecun.com/exdb/mnist/ """

    url = "https://www.cs.toronto.edu/~kriz"
    filename = "cifar-10-python.tar.gz"

    data_path = os.path.abspath(data_path)
    if not pathlib.Path(data_path).exists() and download:
        download_data(os.path.join(url, filename), data_path)
        extract(os.path.join(data_path, filename), data_path)

    filenames = [os.path.join(data_path, "cifar-10-batches-py",
            "data_batch_" + str(i+1)) for i in range(5)]

    images, labels = [], []
    for filename in filenames:
        x, y = unpickle(filename)
        images.append(x)
        labels.append(y)

    x_train = np.concatenate(images, axis=0)
    y_train = np.concatenate(labels, axis=0)

    filename = os.path.join(data_path, "cifar-10-batches-py", "test_batch")
    x_test, y_test = unpickle(filename)
    
    filename = os.path.join(data_path, "cifar-10-batches-py", "batches.meta")
    with open(filename, 'rb') as f:
        meta = pickle.load(f)
    class_names = meta['label_names']

    return (x_train, y_train), (x_test, y_test), class_names


if __name__ == "__main__":

    data_path = "../../datasets/cifar10"
    print(os.path.abspath(data_path))

    train_data, test_data, class_names = load_cifar10(data_path, download=True)

    images, labels = train_data
    print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
    print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())
    print("classes:", class_names, '\n')