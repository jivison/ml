import math

def range_(array):
  return range(len(array))

# Maps x between 0 and one TODO replace with ReLU
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Derivative of the sigmoid
def sigmoidPrime(x):
    Sx = sigmoid(x)
    return Sx * (1 - Sx)

# Object that holds all the secrets to executing backpropagation
class Del_Calc():

    # Stores the history of this object, indexed by: epoch_number, then node
    history = None

    # Stores the delta values to be averaged
    delList = {}

    # Start history 
    def __init__(self, layers):
        self.history = {}


    # Record to history
    def recordNode(self, epoch_number, node, a, b, delB):
      if not self.history[epoch_number]:
        self.history[epoch_number] = {}
      else:
        self.history[epoch_number][node] = {
          "a" : a,
          "b" : b,
          "delB" : delB
        }

    def recordSynapse(self, epoch_number, synapse, w, delW):
      if not self.history[epoch_number]:
        self.history[epoch_number] = {}
      else:
        self.history[epoch_number][synapse] = {
          "w" : w,
          "delW" : delW
        }
      

    # Calculates the rate of change of the bias with respect to the cost
    def delBias(self, layer, prevLayer, advLayer, node, nodeObj, epoch_number, trainingObj=None, node_index=None):

      delBias =  self.calcDelta(
        prevLayer, 
        layer, 
        node,
        nodeObj, 
        advLayer=advLayer if layer != "output" else None,
        output=True if layer == "output" else False,
        trainingObj= trainingObj if layer == "output" else None,
        node_index=node_index if layer == "output" else None
        )
      self.recordNode(epoch_number, node, nodeObj[layer][node]["a"], nodeObj[layer][node]["b"], delBias)
      return delBias


    # Calculates the rate of change of the weight with respect to the cost
    def delWeight(self, layer, prevLayer, synapse, nodeObj, epoch_number, advLayer=None):
      aSource = nodeObj[layer][f"node_{synapse.partition('->')[0]}"]["a"]
      delta = self.calcDelta(prevLayer, layer, f"node_{synapse.partition('->')[0]}", nodeObj, advLayer=advLayer)
      self.recordSynapse(epoch_number, synapse, nodeObj[layer][f"node_{synapse.partition('->')[0]}"]["synapses"][synapse]["w"], aSource * delta)
      return aSource * delta
      
    # Calculates the rate of change of the layer with respect to cost
    def calcDelta(self, prevLayer, layer, node, nodeObj, trainingObj=None, advLayer=None, output=False, node_index=None ):
      wSum = 0
      activation = nodeObj[layer][node]["a"]
      bias = nodeObj[layer][node]["b"]
                
      # Iterate through the synapses, summing the weights of the synapses that influence param node
      for _node, _node_dict in nodeObj[prevLayer].items():
        if _node != "delta":
          for synapse, synapse_dict in _node_dict["synapses"].items():
            
            # Synapse syntax: source_node_id->destination_node_id
            if synapse.partition("->")[2] == node.partition("_")[2]:
              wSum += synapse_dict["w"]
                
        
      # If it's not the output layer
      if not output:
        wSumAdv = 0
        deltaAdv = nodeObj[advLayer]["delta"]

        # Iterate through the synapses, summing the weights of the synapses that are originate from param node
        for synapse in nodeObj[layer][node]["synapses"].values():
          wSumAdv += synapse['w']

        # The equation for any layer but output
        return (wSumAdv * deltaAdv) * sigmoidPrime(wSum * activation + bias)

      # If it's the output layer
      else:
        y = trainingObj.y[node_index]

        # The equation for the output layer 
        return 2 * (activation - y) * sigmoidPrime(wSum * activation + bias)

    def initHistoryDelDict(self, epoch_number):
      self.history[epoch_number] = {"deltaDict" : {}}
      
    def avgDelta(self, layer, epoch_number):
      avg = sum(self.delList[layer]["deltaList"])/len(self.delList[layer]["deltaList"])
      self.history[epoch_number]["deltaDict"][layer] = avg
      return avg
      
    def calcDelAll(self, node_obj, layers, trainingObj, epoch_number):
      """
      Calculates in order:
        1.  delta of output layer; calls avgDelta()
        2.  delta of each node; calls avgDelta() for each previous layer
        3.  delta of each weight
        4.  delta of each bias
      """

      for layer in layers:
        self.delList[layer] = {
          "synapses" : [],
          "biases" : [],
          "deltaList" : []
        }

      output_node_i = 0
      # 1.  delta of output layer; calls avgDelta()
      for node, node_dict in node_obj["output"].items():
       if node != "delta":
        self.delList["output"]["deltaList"].append(self.calcDelta(layers[-1], "output", node, node_obj, trainingObj=trainingObj, output=True, node_index=output_node_i))
        output_node_i += 1
      
      node_obj["output"]["delta"] = self.avgDelta("output", epoch_number)

      hiddenlayers = layers.copy()

      # 2.  delta of each node; calls avgDelta() for each previous layer
      del hiddenlayers[-1] # Removes output from the list of layers as it has already been calculated
      del hiddenlayers[0] # Removes input from the list of layers as it cannot be changed
      



      for layer_i in range_(layers)[::-1]:
        hidden_node_i = 0
        for node in node_obj[layers[layer_i]]:
          if node != "delta":

            self.delList[layers[layer_i]]["deltaList"].append(
              self.calcDelta(
                layers[layer_i - 1], 
                layers[layer_i], 
                node, 
                node_obj, 
                advLayer=layers[layer_i + 1] if layers[layer_i] != "output" else None,
                output=True if layers[layer_i] == "output" else False,
                trainingObj=trainingObj if layers[layer_i] == "output" else None,
                node_index=hidden_node_i
                )
              )
            hidden_node_i += 1
        
        node_obj[ layers[layer_i] ]["delta"] = self.avgDelta(layers[layer_i], epoch_number)


      # 3. & 4.  delta of each weight and each bias
      for layer_i in range_(layers)[::-1]:
        for node, node_dict in node_obj[layers[layer_i]].items():
          if node != "delta":          
            for synapse, synapse_dict in node_dict["synapses"].items():
              self.delList[layers[layer_i]]["synapses"].append({
                synapse : self.delWeight(
                  layers[layer_i], 
                  layers[layer_i - 1], 
                  synapse, 
                  node_obj,
                  epoch_number,
                  advLayer=layers[layer_i + 1] if layers[layer_i] != "output" else None,
                  )
                })
            
            if layers[layer_i] != "input" and layers[layer_i] != "output":
              self.delList[layers[layer_i]]["biases"].append({
                node : self.delBias(layers[layer_i], layers[layer_i - 1], layers[layer_i + 1], node, node_obj, epoch_number)
              })
        
        output_node_i = 0

        for node in node_obj["output"]:
          if node != "delta":
            self.history[epoch_number][node] = {"delB" : node_obj["output"]["delta"]}
            self.delList["output"]["biases"].append({
              node : node_obj["output"]["delta"]
            })
            

      return self.delList