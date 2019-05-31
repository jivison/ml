import json

from main import NETWORK, LAYERS, printNodes

NAME = "badmath"

n = NETWORK()
l = LAYERS()

checkpoints = None

with open(f"checkpoints/{NAME}.json", "r") as jsonfile:
    checkpoints = json.load(jsonfile)


n.input_layer = {"a" : float(input("Enter an input: ").strip())}

l.w1 = checkpoints["a1"]["weight"]
l.w2 = checkpoints["a2"]["weight"]
l.w3 = checkpoints["output"]["weight"]

l.b1 = checkpoints["a1"]["bias"]
l.b2 = checkpoints["a2"]["bias"]
l.b3 = checkpoints["output"]["bias"]

l.output_layer = {"a" : l.calcActivation(l.w3, l.b3, checkpoints["a2"]["activation"])}

printNodes(False)