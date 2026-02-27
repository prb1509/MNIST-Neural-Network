from model import NeuralNetwork
from load_mnist_data import load_images, load_labels
import numpy as np

np.seterr(over='raise')
def main():
    model = NeuralNetwork([784,128,10])
    training_data = load_images("mnist/train-images.idx3-ubyte.zip")
    training_labels = load_labels("mnist/train-labels.idx1-ubyte")
    test_data = load_images("mnist/t10k-images.idx3-ubyte.zip")
    test_labels = load_labels("mnist/t10k-labels.idx1-ubyte")
    model.train(training_data,training_labels,n_epochs=10,learning_rate=0.01)
    model.predict(test_data,test_labels)

main()