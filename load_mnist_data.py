import numpy as np
import zipfile

def load_images(filename, normalize=True):
    with zipfile.ZipFile(filename, "r") as z:
        unzip_file_name = [name for name in z.namelist() if name.endswith("-images.idx3-ubyte")][0]
        with z.open(unzip_file_name) as f:
            magic_bytes = int.from_bytes(f.read(4), byteorder="big")
            n_images = int.from_bytes(f.read(4), byteorder="big")
            n_rows = int.from_bytes(f.read(4), byteorder="big")
            n_cols = int.from_bytes(f.read(4), byteorder="big")
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(n_images, n_rows, n_cols)
            new_images = images.reshape(images.shape[0], n_rows * n_cols)
            if normalize:
                 new_images = new_images / 255
            return new_images


def load_labels(filename):
    with open(filename, "rb") as f:
        magic_bytes = int.from_bytes(f.read(4), byteorder="big")
        num_labels = int.from_bytes(f.read(4), byteorder="big")
        labels = np.frombuffer(f.read(), np.uint8)
    return postprocess_labels(labels)


def postprocess_labels(labels):
    n_labels = labels.shape[0]
    new_labels = np.zeros([n_labels,10,])
    for i in range(n_labels):
        new_labels[i][labels[i]] = 1
    return new_labels