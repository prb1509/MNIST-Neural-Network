import numpy as np
from activations import LeakyReLU, Sigmoid

class NeuralNetwork():
    def __init__(self, layer_sizes, weights=None, biases=None):
        self.n_layers = len(layer_sizes)
        self.weights = []
        for i in range(self.n_layers - 1):
            shape = (layer_sizes[i + 1], layer_sizes[i])
            self.weights.append(self.get_random_weights(shape))
        self.hidden_activation_function = LeakyReLU() 
        self.final_layer_activation = Sigmoid()
        self.activations_cache = []
        self.biases = [np.random.uniform(-0.001,0.001,layer_size) for layer_size in layer_sizes[1:]]
    
    def backpropagate(self, expected_value, learning_rate):
        delta_weights = [np.zeros(weights.shape) for weights in self.weights]
        delta_biases = [np.zeros(biases.shape) for biases in self.biases]

        # Update the weights for the final layer
        # Cost derivative
        cache_value = self.cost_derivative(self.activations_cache[-1], expected_value) 
        delta_weights[-1] = np.outer(cache_value, self.activations_cache[-2]) * learning_rate
        delta_biases[-1] = cache_value * learning_rate

        # Recursively update weights for all other layers backwards
        for i in range(-2, -len(delta_weights) - 1, -1):
            cache_value = self.hidden_activation_function.derivative(self.activations_cache[i]) * (np.transpose(self.weights[i + 1]) @ cache_value)
            delta_weights[i] = np.outer(cache_value, self.activations_cache[i - 1]) * learning_rate
            delta_biases[i] = cache_value * learning_rate

        self.update_weights(delta_weights)
        self.update_biases(delta_biases)


    def forward_propagate(self, input_layer):
        self.activations_cache = []
        activations = input_layer
        self.activations_cache.append(activations)
        for idx, weights in enumerate(self.weights):
            if idx != len(self.weights) - 1:
                activations = self.hidden_activation_function.activation(weights @ activations + self.biases[idx]) 
            else:
                activations = self.final_layer_activation.activation(weights @ activations + self.biases[idx]) 
            self.activations_cache.append(activations)
        return activations


    def train(self, images, labels, learning_rate=0.1, n_epochs=100, dump_values=True):
        for epoch in range(n_epochs):
            images, labels = self.shuffle_data(images,labels)
            for idx, image in enumerate(images):
                label = labels[idx]
                self.forward_propagate(image)
                self.backpropagate(label,learning_rate)
            print(f"--------------------Finished epoch {epoch + 1}--------------------")
        if dump_values:
            print(self.weights)
            print(self.biases)
            print(self.activations_cache)


    def save_model(self, filename="idk one day I'll write this"):
        pass
    

    def predict(self, images, labels):
        total = 0
        correct = 0
        for idx, image in enumerate(images):
            total += 1
            label = labels[idx]
            self.forward_propagate(image)
            prediction = np.argmax(self.activations_cache[-1])
            if prediction == np.where(label == 1)[0][0]:
                correct += 1
        print(correct / total)


    def update_weights(self, weights):
        for idx, weight in enumerate(weights):
            self.weights[idx] -= weight


    def update_biases(self, biases):
        for idx, bias in enumerate(biases):
            self.biases[idx] -= bias


    def cost_derivative(self, activations, expectation):
        # Assuming cross entropy cost function and sigmoid final layer activation
        # This is actually derivative of cost function x derivative of final layer activation
        return activations - expectation


    @staticmethod
    def load_model(filename="tbd"):
        pass
        

    @staticmethod
    def get_random_weights(shape):
        limit = np.sqrt(6 / (sum(shape)))
        return np.random.uniform(-limit, limit, size=shape)


    @staticmethod
    def shuffle_data(data, labels):
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        return data[indices], labels[indices]