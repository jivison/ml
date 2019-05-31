import random
import math
import os
import time
import json



# TODO
# Fix delCalc so it takes average
# http://neuralnetworksanddeeplearning.com/chap2.html#eqtnBP1


NAME = "languageGuesser"
# EPOCHS = 5000
LEARNING_RATE = 0.5

# Desired output of the network
import subfuncs.trainer as trainer

# Calculates cost
import subfuncs.cost as cost

# Calulates change
import subfuncs.del_calc as del_calc

import nodes as nodes

# Initialize Cost object
COST = cost.Cost()

# Initialize Desired object
TRAINER = trainer.Trainer()

# Initialize Nodes object
NODES = nodes.Nodes("params.json")

# Initialize Del_Cal object
DEL_CALC = del_calc.Del_Calc(NODES.layers)

# Maps between zero and one
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

            
class LAYERS():

    random = random.randint(0, 7)

    def __init__(self):
        NODES.intialize([x for x in range(10)])
        self.nodes = NODES.nodes
        self.layers = NODES.layers
        

    def calcActivationSum(self, wL, aPrev):
        return wL * aPrev

    def desired_output(self, output_index):
        if output_index == self.random:
            return 1
        else:
            return 0

    def calcActivationSynapse(self, layer, synapseName):
        sourceNode = self.nodes[layer][f"node_{synapseName.partition('->')[0]}"]
        aSource = sourceNode["a"]
        weight = sourceNode["synapses"][synapseName]["w"]
        
        return self.calcActivationSum(weight, aSource)

    def calcActivationAll(self):
        for layer_i in range(len(self.layers)):
            layer = self.layers[layer_i]

            destination_dict = {}

            if layer != "output":
                for node, node_dict in self.nodes[layer].items():
                    if node != "delta":
                        for synapse in node_dict["synapses"]:

                            destination_node = synapse.partition("->")[2]

                            try:
                                destination_dict[destination_node] += self.calcActivationSynapse(layer, synapse)
                            except KeyError:
                                destination_dict[destination_node] = self.calcActivationSynapse(layer, synapse)

            # print(self.nodes)

            for node, activation in destination_dict.items():
                bias = self.nodes[self.layers[layer_i + 1]][f"node_{node}"]["b"]
                self.nodes[self.layers[layer_i + 1]][f"node_{node}"]["a"] = sigmoid(activation + bias)

    # y is desired output
    def calcCostK(self, a, y):
        return COST.costK(a, y)

    # def calcCostAll(self, epoch_number, cost_obj):
    #     index = 0
    #     cost = 0

    #     for node, node_dict in self.nodes["output"].items():
    #         cost += self.calcCostK(node_dict["a"], self.desired_output(index))
    #         index += 1
             
    #     cost_obj.record(cost/index, epoch_number)        
    #     return cost/index + 1

    # y is desired output
    def calcDelK(self, aPrev, a, wL, bL, y, epoch_number, node_number):
        output = DEL_CALC.calc(aPrev, a, wL, bL, y, epoch_number, node_number)
        return output["weight"], output["bias"]

    def calcDelAll(self, epoch_number):
        return DEL_CALC.calcDelAll(self.nodes, self.layers, TRAINER, epoch_number)

    def applyDelAll(self, del_dict):
        for layer in self.layers:
            d = del_dict[layer]

            for synapse_dict in d["synapses"]:


                for synapse, delweight in synapse_dict.items():
                    sourceNode = "node_" + synapse.partition("->")[0]
                    allSynapses = []
                    for epoch_number in DEL_CALC.history:
                        allSynapses.append(DEL_CALC.history[epoch_number][synapse]["delW"])
                    self.nodes[layer][sourceNode]["synapses"][synapse]["w"] += sum(allSynapses)/len(allSynapses)

            for node_dict in d["biases"]:
                for node, delbias in node_dict.items():
                    allNodes = []
                    for epoch_number in DEL_CALC.history:
                        allNodes.append(DEL_CALC.history[epoch_number][node]["delB"])
                    if node != "delta":
                        self.nodes[layer][node]["b"] += sum(allNodes)/len(allNodes)

    def defInitNodes(self, dataset):
        index = 0
        # Generate input nodes
        for node, node_dict in self.nodes["input"].items():       
            if node != "delta":  
                node_dict["a"] = dataset[index]
                index += 1
                    

l = LAYERS()


score = []

def epoch(learning_rate, epoch_number):

    os.system("clear")

    print("EPOCH " + str(epoch_number))

    expectedOutputNodes, generatedInputNodes = TRAINER.epoch()

    TRAINER.y = expectedOutputNodes

    l.defInitNodes(generatedInputNodes)

    # Calulate and apply the activations for all non-input layers
    l.calcActivationAll()

    networkOutput = []
    
    for node, node_dict in l.nodes["output"].items():
        if node != "delta":
            networkOutput.append(node_dict["a"])

    print(f"Input: {generatedInputNodes}\nNetwork Output: {networkOutput}\nExpected Output: {expectedOutputNodes}")

    cost = l.calcCostK(networkOutput[0], expectedOutputNodes[0])

    print(f"\nCost:{cost}")

    if cost < 0.01:
        score.append(1)
    else:
        score.append(0)

    print(f"\nNetwork guessed: {'Correct!' if cost < 0.01 else 'Wrong!'}")

    
    print(f"Score: {round(sum(score)/len(score) * 100)}%")

    # # Network architecture
    # for layer in l.layers:
    #     print(layer)
    #     for node in l.nodes[layer]:
    #         print("\t" + node)

    # for layer in l.layers:
    #     print(layer)
    #     for node, node_dict in l.nodes[layer].items():
    #         print(f"\t{node}")
    #         for key, value in node_dict.items():
    #             print(f"\t\t{key} : {value}")

    # Calculate the cost
    # l.cost = l.calcCostAll(i, COST)

    # Initialize the delta history entry for this epoch
    DEL_CALC.initHistoryDelDict(epoch_number)

    # Generate the list of changes to weights and biases
    deldict = l.calcDelAll(epoch_number)

    # Apply the changes
    l.applyDelAll(deldict)





index = 1
proceed = "s"
reqIndex = 0

while True:
    epoch(LEARNING_RATE, index)
    index += 1
    

    if proceed == "s" or proceed == "":
        proceed = input("[s]ingle, [<n>] number of steps: ")
        if proceed != "s" and proceed != "":
            reqIndex = int(proceed)
            


    if proceed != "s" and proceed != "":
        if index % reqIndex == 0:
            proceed = "s"