import json
import math
import random

# For naming the nodes
letters = list("abcdefghijklmnopqrstuvwxyz")

class Nodes():
    # Dict object containing all the nodes
    nodes = {}

    # List holding all the layers, makes things a lot easier
    layers = []

    # Global letter index to keep consistency between layers
    current_letter_index = 0

    # paramsfile needs to be in json format
    def __init__(self, paramsfile):
        with open(paramsfile, "r") as params:
            
            # Store the parameters in the Nodes object
            self._params = json.load(params)

            # Add all the layers
            self.layers.append("input")
            for hidden in range(len(self._params["hidden_layers"]["nodes"])):
                self.layers.append(f"hidden_{hidden}")
            self.layers.append("output")

            # Generate names for all the nodes according to the parameters
            self.index(self._params)
            
            # Generate weights and biases for each node
            self.generateSynapses(self._params)
        
    def generateSynapses(self, params_dict):
        
        for layer_i in range(len(self.layers) - 1):
            current_layer = self.layers[layer_i]
            next_layer = self.layers[layer_i + 1]
                
            for cnode, cnode_dict in self.nodes[current_layer].items():
                
                cnode_name = self.separateNodeName(cnode)

                self.nodes[current_layer][cnode]["b"] = random.uniform(*self._params["hidden_layers"]["bounds"]["bias"])

                for nnode, nnode_value in self.nodes[next_layer].items():

                    nnode_name = self.separateNodeName(nnode)
                    
                    self.nodes[current_layer][cnode]["synapses"][self.generateSynapseName(cnode_name, nnode_name)] = {
                            "w" : random.uniform(*self._params["hidden_layers"]["bounds"]["weight"]),
                        }
        
        for node, node_dict in self.nodes["output"].items():
            node_dict["b"] = random.uniform(*self._params["hidden_layers"]["bounds"]["bias"])
                    
    def intialize(self, dataset):
        index = 0
        random.shuffle(dataset)
        # Generate input nodes
        for node, node_dict in self.nodes["input"].items():
            # print(node_dict)
            node_dict["a"] = dataset[index]
            index += 1
                        


    def index(self, params_dict):
        input_params = params_dict["input_layer"]
        hidden_params = params_dict["hidden_layers"]
        output_params = params_dict["output_layer"]

        # Initializing the lists that will contain the nodes in each layer
        self.nodes["input"] = {}
        self.nodes["output"] = {}
        # hidden layers inited in the for loop

        # Generate nodes for input layer
        for i in range(input_params["number of nodes"]):
            self.nodes["input"][f"node_{self.getLetter(self.current_letter_index)}"] = {
                "synapses": {}
                }

            self.current_letter_index += 1
        
        # Generate nodes for each hidden layer
        for hidden_layer in range(len(hidden_params["nodes"])):
            self.nodes[f"hidden_{hidden_layer}"] = {}
            for i in range(hidden_params["nodes"][hidden_layer]):
                self.nodes[f"hidden_{hidden_layer}"][f"node_{self.getLetter(self.current_letter_index)}"] = {
                "synapses": {}
                }

                self.current_letter_index += 1

        # Generate nodes for output layer 
        for i in range(output_params["number of nodes"]):
            self.nodes["output"][f"node_{self.getLetter(self.current_letter_index)}"] = {
                "synapses": {}
                }
            
            self.current_letter_index += 1

    def getLetter(self, index):

        # Returns letter based on a, b, c, ...y, z, aa, ab, ac
        if index > 25:
            return letters[math.floor(index/26) - 1] + letters[(index % 26)]
        else:
            return letters[(index % 26)]

    def separateNodeName(self, nodeName):
        return nodeName.partition("_")[2]

    def generateSynapseName(self, node1, node2):
        return f"{node1}->{node2}"
            

# n = Nodes("params.json")

# print(n.nodes)

