import random
import math
import os
import time
import json


NAME = "badmath"
EPOCHS = 5000
LEARNING_RATE = 0.5

# Desired output of the network
import subfuncs.desired_output as desired

# Calculates cost
import subfuncs.cost as cost

# Calulates change
import subfuncs.del_calc as del_calc

# Initialize Cost object
COST = cost.Cost()

# Initialize Desired object
DESIRED = desired.Desired()

# Initialize Del_Cal object
DEL_CALC = del_calc.Del_Calc()

# Maps between zero and one
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class NETWORK():
    
    def __init__(self):
        self.input_layer = {"a" : random.uniform(-15, 15)}
        self.desired_output = DESIRED.get_value()
        self.cost = None
        
class LAYERS(NETWORK):

    def __init__(self):
        # LAYER 1
        self.w1 = random.uniform(-1, 1)
        self.b1 = random.uniform(-10, 10)
        self.a1 = None
        self.delw1 = None
        self.delb1 = None

        # LAYER 2
        self.w2 = random.uniform(-1, 1)
        self.b2 = random.uniform(-10, 10)
        self.a2 = None   
        self.delw2 = None
        self.delb2 = None     

        # OUTPUT LAYER
        self.w3 = random.uniform(-1, 1)
        self.b3 = random.uniform(-10, 10)
        self.delw3 = random.uniform(-1, 1)
        self.delb3 = random.uniform(-10, 10)
        self.output_layer = {"a" : None}

    def calcActivation(self, wL, bL, aPrev):
        return sigmoid(wL * aPrev + bL)
    
    # y is desired output
    def calcCostK(self, a, y, epoch_number):
        return COST.costK(a, y, epoch_number)

    # y is desired output
    def calcDel(self, aPrev, a, wL, bL, y, epoch_number, node_number):
        output = DEL_CALC.calc(aPrev, a, wL, bL, y, epoch_number, node_number)
        return output["weight"], output["bias"]

n = NETWORK()
l = LAYERS()

def printNodes(refresh = False, keyword = None, keyword2 = None):
    l.nodes = {
                "a1" : {
                    "weight" : l.w1,
                    "bias" : l.b1,
                    "activation" : l.a1,
                    "delweight" : l.delw1,
                    "delbias" : l.delb1,
                },
                "a2" : {
                    "weight" : l.w2,
                    "bias" : l.b2,
                    "activation" : l.a2,
                    "delweight" : l.delw2,
                    "delbias" : l.delb2,
                },
                "output" : {
                    "weight" : l.w3,
                    "bias" : l.b3,
                    "activation" : l.output_layer["a"],
                    "delweight" : l.delw3,
                    "delbias" : l.delb3,
                }
            }

    if not refresh:
        for node, node_dict in l.nodes.items():
            if not keyword:
                print(f"\t{node}")
                for key, item in node_dict.items():
                    if not keyword2:
                        print(f"\t{key}: {item}")
                    if key == keyword2:
                        print(f"\t{key}: {item}")
            if node == keyword:
                print(f"\t{node}")
                for key, item in node_dict.items():
                    if not keyword2:
                        print(f"\t{key}: {item}")
                    if key == keyword2:
                        print(f"\t{key}: {item}")
                
def epoch(learning_rate, epoch_number):

    l.a1 = l.calcActivation(l.w1, l.b1, n.input_layer["a"])
    l.a2 = l.calcActivation(l.w2, l.b2, l.a1)
    l.output_layer["a"] = l.calcActivation(l.w3, l.b3, l.a2)
    
    # n.cost = l.calcCostK(l.nodes["output"]["activation"], n.desired_output)
    # print(f"Cost: {n.cost}")


    l.delw3, l.delb3 = l.calcDel(l.a2, l.output_layer["a"], l.w3, l.b3, n.desired_output, epoch_number, "3")


    l.delw2, l.delb2 = l.calcDel(l.a1, l.a2, l.w2, l.b2, n.desired_output, epoch_number, "2")


    l.delw1, l.delb1 = l.calcDel(n.input_layer["a"], l.a1, l.w1, l.b1, n.desired_output, epoch_number, "1")

    l.w1 -= l.delw1 * learning_rate
    l.w2 -= l.delw2 * learning_rate
    l.w3 -= l.delw3 * learning_rate

    l.b1 -= l.delb1 * learning_rate
    l.b2 -= l.delb2 * learning_rate
    l.b3 -= l.delb3 * learning_rate

    n.cost = l.calcCostK(l.output_layer["a"], n.desired_output, epoch_number)
    
    print(f"\tcost: {n.cost}")

    printNodes(False, "output", "activation")

printNodes(True)

epoch(LEARNING_RATE, 0)

for i in range(EPOCHS - 1):

    initial_a = l.nodes['output']['activation']

    epoch(LEARNING_RATE, i + 1)

    # os.system("clear")
    print(i)
    print(f"\tdelta: {100 - initial_a/l.nodes['output']['activation'] * 100}", flush=True)

    
    
with open(f"checkpoints/{NAME}.json", "w+") as jsonfile:
    json.dump(l.nodes, jsonfile)
