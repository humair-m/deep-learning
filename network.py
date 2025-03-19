"""
network.py
~~~~~~~~~~
#ALL RIGHTS RESERVERD
#HUMAIR MUNIR
#humairmunirawan@gmail.com

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that this is a modernized version of the
code from the book "Neural Networks and Deep Learning" by Michael Nielsen.
"""

import numpy as np
import random


class Network:
    def __init__(self, sizes):
        """
        Initialize a Neural Network model.
        
        Parameters:
        sizes : list of int
            The number of neurons in the respective layers of the network.
            For example, if the list was [2, 3, 1] then it would be a three-layer network,
            with the first layer containing 2 neurons, the second layer 3 neurons,
            and the third layer 1 neuron.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases and weights with random values
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        """
        Return the output of the network if `a` is input.
        
        Parameters:
        a : numpy.ndarray
            Input activation for the network.
            
        Returns:
        numpy.ndarray
            Output activation of the network.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None, early_stopping=False, patience=5):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        
        Parameters:
        training_data : list of tuples (x, y)
            Training inputs and corresponding desired outputs.
        epochs : int
            Number of epochs to train for.
        mini_batch_size : int
            Size of mini-batches for stochastic gradient descent.
        learning_rate : float
            Learning rate for gradient descent.
        test_data : list of tuples (x, y), optional
            If provided, the network will be evaluated against the test data
            after each epoch, and progress will be printed out.
        early_stopping : bool, optional
            Whether to use early stopping to prevent overfitting.
        patience : int, optional
            Number of epochs to wait for improvement before stopping if early_stopping is True.
        """
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            best_accuracy = 0
            no_improvement_count = 0
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            if test_data:
                accuracy = self.evaluate(test_data) / n_test
                print(f"Epoch {j+1}: {accuracy * 100:.2f}% accuracy on test data")
                
                if early_stopping:
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    if no_improvement_count >= patience:
                        print(f"Early stopping at epoch {j+1}")
                        return
            else:
                print(f"Epoch {j+1} complete")
    
    def update_mini_batch(self, mini_batch, learning_rate):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch.
        
        Parameters:
        mini_batch : list of tuples (x, y)
            Mini batch of training data.
        learning_rate : float
            Learning rate for gradient descent.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        batch_size = len(mini_batch)
        self.weights = [w - (learning_rate / batch_size) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / batch_size) * nb
                      for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        """
        Return a tuple `(nabla_b, nabla_w)` representing the gradient
        for the cost function with respect to the biases and weights.
        
        Parameters:
        x : numpy.ndarray
            Input activation.
        y : numpy.ndarray
            Desired output activation.
            
        Returns:
        tuple of lists
            nabla_b: list of numpy arrays, gradients for biases
            nabla_w: list of numpy arrays, gradients for weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Forward pass
        activation = x
        activations = [x]  # list to store all activations, layer by layer
        zs = []  # list to store all z vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Note that the variable l in the loop below is used differently
        # than in the book. Here, l = 1 means the last layer of neurons,
        # l = 2 is the second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return nabla_b, nabla_w
    
    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network
        outputs the correct result.
        
        Parameters:
        test_data : list of tuples (x, y)
            Test inputs and corresponding desired outputs.
            
        Returns:
        int
            Number of correct predictions.
        """
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                         for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives of the cost function
        with respect to the output activations.
        
        Parameters:
        output_activations : numpy.ndarray
            Output activations of the network.
        y : numpy.ndarray
            Desired output activations.
            
        Returns:
        numpy.ndarray
            Vector of partial derivatives.
        """
        return output_activations - y
    
    def save(self, filename):
        """
        Save the neural network to a file.
        
        Parameters:
        filename : str
            Name of the file to save the network to.
        """
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        np.save(filename, data)
    
    @classmethod
    def load(cls, filename):
        """
        Load a neural network from a file.
        
        Parameters:
        filename : str
            Name of the file to load the network from.
            
        Returns:
        Network
            A neural network instance.
        """
        data = np.load(filename, allow_pickle=True).item()
        net = cls(data["sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net


# Helper functions
def sigmoid(z):
    """
    The sigmoid activation function.
    
    Parameters:
    z : numpy.ndarray
        Input to the sigmoid function.
        
    Returns:
    numpy.ndarray
        Output of the sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    
    Parameters:
    z : numpy.ndarray
        Input to the sigmoid function.
        
    Returns:
    numpy.ndarray
        Derivative of the sigmoid function.
    """
    return sigmoid(z) * (1 - sigmoid(z))
