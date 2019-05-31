from networkParams import params

import numpy as np

def nodes(params):
    weights = {}
    biases = {}

    # Create the the list of layers
    layers = ["input"]
    for layer in range(len(params["hidden"])):
        layers.append(f"hidden_{layer}")
    layers.append("output")

    # Create the weight matrices for each layer-layer connection
    for layer_i in range(len(layers) - 1):
        layer = layers[layer_i]
        advLayer = layers[layer_i + 1]

        weights[f"{layer}->{advLayer}"] = np.random.rand(params["input_nodes"], 1)

    

        









    return weights, biases


print(nodes(params))