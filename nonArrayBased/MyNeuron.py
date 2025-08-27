class neuron:
    inWeights = [] #weights on input
    inNeuron = []
    outWeights = [] #weights applied to its output to each output neuron
    outNeuron = []
    length = 0
    bias = 0
    lastInput = []

    def __init__(self, weights, length, bias):
        self.inWeights = weights
        self.length = length
        self.bias = bias

    def getBias(self):
        return self.bias
    
    def setBias(self, bias):
        self.bias = bias

    def getInNeuron(self):
        return self.inNeuron
    
    def setInNeuron(self, neurons):
        self.inNeuron = neurons

    def getOutNeuron(self):
        return self.outNeuron
    
    def setOutNeuron(self, neurons):
        self.outNeuron = neurons

    def getInWeights(self):
        return self.inWeights
    
    def setInWeights(self, weights):
        self.inWeights = weights

    def getOutWeights(self):
        return self.outWeights

    def setOutWeights(self, weights):
        self.outWeights = weights

    def sumWeights(self, inputs):
        out = 0
        for i in range(self.length):
            out += inputs[i] * self.inWeights[i]
        #print(out)
        return out + self.bias
    
    def getLastInput(self):
        return self.lastInput

    def evaluateNeuron(self, inputs, function):
        self.lastInput = inputs
        #print("After activation function: ", function(self.sumWeights(inputs)))
        return function(self.sumWeights(inputs))