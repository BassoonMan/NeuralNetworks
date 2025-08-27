from nonArrayBased.MyNeuron import neuron as Neuron

@staticmethod
def default(input): # functional
    return input

class layer:
    neurons = []
    neuronLength = 0
    activationFunc = default

    def __init__(self, neurons, function=None):
        self.neurons = neurons
        self.neuronLength = len(neurons)
        if function != None:
            self.activationFunc = function

    def getLength(self):
        return self.neuronLength
    
    def setActivationFunc(self, function):
        self.activationFunc = function

    def getActivationFunc(self):
        return self.activationFunc

    def addNeuron(self, neuron):
        self.neurons.append(neuron)
        self.neuronLength += 1

    def getNeuron(self, index):
        if index > self.neuronLength:
            return None
        return self.neurons[index]
    
    def getNeurons(self):
        return self.neurons
    
    def evaluateLayer(self, inputs):
        outputs = []
        #i = 0
        for neuron in self.neurons:
            #print("Neuron in layer: ", i)
            outputs.append(neuron.evaluateNeuron(inputs, self.activationFunc))
            #i += 1
        return outputs