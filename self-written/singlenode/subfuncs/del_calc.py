import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoidPrime(x):
    Sx = sigmoid(x)
    return Sx * (1 - Sx)

class Del_Calc():

    history = None

    def __init__(self):
        self.history = {}
        pass

    def record(self, epoch_number, node_number, a, wL, bL, delB, delW):
        try: 
          self.history[epoch_number][node_number] = {
            "weight" : wL,
            "bias" : bL,
            "activation" : a,
            "delbias" : delB,
            "delweight" : delW
          }
        except KeyError:
          self.history[epoch_number] = {}
          self.history[epoch_number][node_number] = {
            "weight" : wL,
            "bias" : bL,
            "activation" : a,
            "delbias" : delB,
            "delweight" : delW
          }


    def calc(self, aPrev, a, wL, bL, y, epoch_number, node_number):
        bias = sigmoidPrime(wL * aPrev * + bL) * 2 * (a - y)
        weight = aPrev * bias
        self.record(str(epoch_number), str(node_number), a, wL, bL, bias, weight)
        # weights = []
        # biases = []
        # for key, value in self.history.items():
        #   weights.append(value[node_number]["delweight"])
        #   biases.append(value[node_number]["delbias"])

        # return {"bias" : sum(biases)/len(biases), "weight" : sum(weights)/len(weights)}
        
        # The average yields lower cost
        return {"bias" : bias, "weight" : weight}