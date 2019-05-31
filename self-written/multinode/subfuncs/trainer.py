import random


dataset = [
    {"in" : [0, 0],
    "out" : [0]},
    {"in" : [0, 1],
    "out" : [1]},
    {"in" : [1, 0],
    "out" : [1]},
    {"in" : [1, 1],
    "out" : [0]},
]

class Trainer():
    
    def __init__(self):
        pass

    def epoch(self):
        tmp = dataset[random.randint(0, len(dataset)) - 1]
        return tmp["out"], tmp["in"]