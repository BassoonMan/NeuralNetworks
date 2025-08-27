from nonArrayBased.MyNeuron import neuron as Neuron
from nonArrayBased.MyLayer import layer as Layer

class network:
    layers = [] # list of lists indexed by layer
    layerLength = 0

    # def __init__(self, layers):
    #     self.layers = layers
    #     self.layerLength = len(layers)

    def addLayer(self, layer):
        self.layers.append(layer)
        self.layerLength += 1
        if self.layerLength == 1:
            for i in range(self.layers[0].getLength()):
                self.layers[0].getNeuron(i).setOutNeuron(None)
                self.layers[0].getNeuron(i).setInNeuron(None)
            return
        for i in range(self.layers[-2].getLength()):
            self.layers[-2].getNeuron(i).setOutNeuron(self.layers[-1].getNeurons())
        for i in range(self.layers[-1].getLength()):
            self.layers[-1].getNeuron(i).setInNeuron(self.layers[-2].getNeurons())
            self.layers[-1].getNeuron(i).setOutNeuron(None)

    def getLayer(self, index):
        if index > self.layerLength:
            return None
        return self.layers[index]
    
    def evaluateNetwork(self, inputs):
        temp = inputs
        for layer in self.layers:
            temp = layer.evaluateLayer(temp)
        return temp
    
    def printInWeights(self):
        for i in range(self.layerLength):
            print(i, "th layer: ")
            for j in range(self.getLayer(i).getLength()):
                print(self.getLayer(i).getNeuron(j).getInWeights(), self.getLayer(i).getNeuron(j).getBias())
    
    def updateNetworkDelta(self, diffs, inputs):
        for i in range(self.getLayer(0).getLength()):
            currNeuron = self.getLayer(0).getNeuron(i)
            weights = currNeuron.getWeights()
            change = [1*diffs[i]*inputs[j] for j in range(len(weights))]
            newWeights = [weights[j]+change[j] for j in range(len(weights))]
            currNeuron.setBias(currNeuron.getBias() + diffs[i])
            currNeuron.setWeights(newWeights)
        return

    def updateNetworkGeneralizedDelta(self, diffs, a, derivativeFunc): # Doesnt Work, IDK why
        newDeltas = []
        lastDeltas = []
        changeManifold = []
        for i in range(self.layerLength):
            lastDeltas = newDeltas
            newDeltas = []
            layerChanges = []
            #print("Current Layer: ", self.layerLength - i - 1)
            currLayer = self.getLayer(self.layerLength - i - 1)
            for j in range(currLayer.getLength()):
                #print("Current Neuron in layer: ", j)
                currNeuron = currLayer.getNeuron(j)
                outNeurons = currNeuron.getOutNeuron()
                inputs = currNeuron.getLastInput()
                sum = currNeuron.sumWeights(inputs)
                #print("sum of weighted inputs: ", sum)
                derivitive = derivativeFunc(sum)
                #print("calculated derivative: ", derivitive)
                #print("last input for this neuron: ", inputs)
                if i != 0:
                    outWeights = [outNeurons[k].getInWeights()[j] for k in range(len(outNeurons))]
                    #print("outfacing weights:", outWeights)
                    delta = 0
                    for k in range(len(lastDeltas)):
                        delta += lastDeltas[k]*outWeights[k]
                    delta *= derivitive
                else:
                    delta = diffs[j] * derivitive
                #print("calculated error signal: ", delta)
                change = [a * delta * inputs[k] for k in range(len(inputs))]
                change.append(a * delta)
                #print("calculated change: ", change)
                newDeltas.append(delta)
                layerChanges.append(change)
            changeManifold.append(layerChanges)

            #print("New deltas from this layer: ", newDeltas)
            #print()
        #print("The changes: ", changeManifold)
        for i in range(self.layerLength):
            currLayer = self.getLayer(self.layerLength - i - 1)
            for j in range(currLayer.getLength()):
                currNeuron = currLayer.getNeuron(j)
                weights = currNeuron.getInWeights()
                change = changeManifold[i][j]
                newWeights = [weights[k] + change[k] for k in range(len(weights))]
                currNeuron.setInWeights(newWeights)
                currNeuron.setBias(currNeuron.getBias() + change[-1])
        return
    
    def updateNetworkGradientDescent():
        pass

    def updateNetworkDeltaNonLinear(self, diffs, inputs, derivativeFunc):
        for i in range(self.getLayer(0).getLength()):
            currNeuron = self.getLayer(0).getNeuron(i)
            weights = currNeuron.getWeights()
            sum = currNeuron.sumWeights(inputs)
            derivitive = derivativeFunc(currNeuron.getActivationFunc(), sum)
            delta = [1*diffs[i]*inputs[j]*derivitive for j in range(len(weights))]
            newWeights = [weights[j]+delta[j] for j in range(len(weights))]
            currNeuron.setWeights(newWeights)
        return