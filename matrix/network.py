# TODO

# Add layer and node customization
# Write infrastructure around guessing languages
    # Generate dataset
    # Contextualize output and input nodes
# Add something?

import numpy as np
import math
import os
from scipy.special import expit

import nodes

# Generates inputs and outputs
import trainer as t

# Holds the parameters for the network
from networkParams import params

# Maps between zero and one
def sigmoid(x):
  return expit(x)

# Derivative of the sigmoid
def sigmoid_prime(x):
    Sx = sigmoid(x)
    return Sx * (1 - Sx)

class NeuralNetwork():
    def __init__(self):
        pass
    
    def initialize(self, x, y):
        self.weights, self.biases = nodes.nodes()

        self.input = x
        # shape returns the dimensions of the array, in this case, the number of rows
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)


    def feedforward(self):
        # the dot product is the sum of all the weights times the inputs
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backpropagate(self, learning_rate):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_prime(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_prime(self.output), self.weights2.T) * sigmoid_prime(self.layer1)))

        self.weights1 += d_weights1 * learning_rate
        self.weights2 += d_weights2 * learning_rate
    
    def train(self, X, y):
        self.output = self.feedforward()
        self.backpropagate(params["learning_rate"])

nn = NeuralNetwork()
trainer = t.Trainer()

X=np.array(([0,0],[0,1],[1,0],[1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

nn.initialize(X, y)

def epoch():

    nn.train(X, y)

    nn.cost = np.mean(np.square(nn.y - nn.output))

    print(f"""
Input: 
{X}
    
Network Output: 
{nn.output}
Expected Output: 
{y}

Cost: {nn.cost}

Network guessed: {"Correct!" if nn.cost < 0.1 else "Wrong!"}
    """)

reqIndex = 0
proceed = "s"
index = 1

while True:
    os.system("clear")
    print(f"EPOCH {index}")
    epoch()

    if proceed == "s" or proceed == "":
        proceed = input("[s]ingle, [<n>] number of steps: ")
        if proceed != "s" and proceed != "":
            reqIndex = int(proceed)
            


    if proceed != "s" and proceed != "":
        if reqIndex == 0:
            proceed = "s"

    reqIndex -= 1
    index += 1
