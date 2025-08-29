import numpy as np
import random as rand
from typing import List

class ArrayNetworkFeedforward:
    """ 
    Class for a neural network implemented using arrays
    ...

         
    Attributes:
    -----------
    weightsByLayer : 2D double array
        Array of weights for the neural network, each column being weights for a layer
    layerLengths : int array
        Array of how many neurons are in each layer
    layers : int
        How many layers there are
    biasByLayer : 2D double array
        Array for the bias for each neuron in the network, each column a layer, each row in a column a neurons bias
    bias : bool
        Keeps track of whether the network has a bias
    layerAct : func Array
        Each layers activation function

    Methods:
    --------
    evaluateNetwork(inputs)
        Evaluates the network for a given input
    printNetwork()
        Prints the weights and biases of the network out
    updateNetworkGeneralizedDelta(self, inputs, diffs, learnRate)
        Updates the network weights using gradient descent
    """
    weightsByLayer = [] # Stored as (nextlayerDim X PreviousLayerDim)
    layerLengths = []
    layers = 0
    biasByLayer = [] # Should be (1 X LayerDim)
    outputsByLayerNonActivated = [] # Should always be stored as an array of (1 X layerDim)
    outputsByLayerActivated = [] # Should always be stored as an array of (1 X layerDim)
    bias = True
    layerAct = [] # List of activation functions and their derivatives
    
    #
    def __init__(self, inputCount, layers, neuronPerLayer, layerAct, random = False, bias = True):
        """ Sets up the network, randomizing the initial weights
         
        Parameters:
        -----------
            inputCount : int
                Length of Input vector
            layers : int
                Number of layers in network
            neuronPerLayer : int array
                Array with each index specifying how many neurons are in that indicies layer
            random : bool
                Switches between randomized initial weights (true) and 0 initialization (false)
            bias : bool
                Switches between the network having a bias on each neuron or not (true or false)
        """
        self.bias = bias
        for i in range(layers):
            if (i == 0):
                n = neuronPerLayer[0]
                m = inputCount
                # First layer depends on input count and point to the first layer length
            else:
                n = neuronPerLayer[i]
                m = neuronPerLayer[i-1]
                # Subsequent layers take previous layer length and point to next layer length
            if (bias):
                bias = []
                for j in range(neuronPerLayer[i]):
                    if (random):
                        bias.append(rand.uniform(-.8,.8))
                    else:
                        bias.append(0)
                self.biasByLayer.append(np.atleast_2d(bias))
                # If bias is turned on add a random bias for each neuron for each layer
            weights = np.zeros((m, n))
            if (random):
                for j in range(n):
                    for k in range(m):
                        weights[k][j] = rand.uniform(-.8, .8)
            # If random is turned on give each layer's weights a random value
            self.weightsByLayer.append(weights)
            self.layerLengths.append(n)
        self.weightsByLayer.append(0)
        self.layers = layers
        self.layerAct = layerAct

    def evaluateNetwork(self, inputs:List[float]):
        """ Evaluates the network for a given input
         
        Parameters:
        -----------
            inputs : array(float)
                Vector of input values
        """
        self.outputsByLayerActivated=[]
        self.outputsByLayerNonActivated=[]
        if (inputs.shape[1] != self.weightsByLayer[0].shape[0]):
            print("evaluateNetwork: Wrong number of inputs!")
            print(str(inputs.shape[1]) + " Vs " + str(self.weightsByLayer[0].shape[0]))
            return None
        temp = inputs # Inputs are given as (1XDim) vectors
        # Sets temp to start as a numpy vector version of the inputs
        #sig = np.vectorize(sigmoid)
        for i in range(self.layers):
        # For each layer:
            if self.bias:
                temp = np.add(np.matmul(temp, self.weightsByLayer[i]), self.biasByLayer[i])
            else:
                temp = np.matmul(temp, self.weightsByLayer[i])
            # Multiply the current vector by this layers weights, adding the bias if turned on
            # Apply the activation function
            act = np.vectorize(self.layerAct[i][0])
            self.outputsByLayerNonActivated.append(temp)
            temp = act(temp)
            self.outputsByLayerActivated.append(temp)
            # Append the outputs to the outputsByLayerNonActivated for use in backpropagation
        return temp
    
    def printNetwork(self):
        """ Prints the weights and biases of the network out
        """
        for i in range(self.layers):
            print("The", i, "th layer has weights: ")
            print(self.weightsByLayer[i])
            if self.bias:
                print("The biases on layer", i, "are", self.biasByLayer[i])

    def outputEmbedding(self):
        """ Prints the weights and biases of the network out
        """
        with open("wordEmbedding.txt", "w") as txt_file:
            for line in self.weightsByLayer[0]:
                txt_file.write(str(line.tolist()) + ",\n") # works with any number of elements in a line
            txt_file.close()
        return self.weightsByLayer[0]

    def updateNetworkGeneralizedDelta(self, inputs, diffs, learnRate):
        """ Updates the network weights using gradient descent
         
        Parameters:
        -----------
            inputs : double array
                Vector of input values, (1XD)
            diffs : double array
                Vector corresponding to the difference between target and outputs, (1XD)
            learnRate : double
                Proportionality constant for how much the change to the weights should be
        """
        changeManifold = [] # Array of all the weight changes
        biasManifold = [] # Array of all the bias changes
        for i in range(self.layers):
            outputs = self.outputsByLayerNonActivated[self.layers - i - 1] # Set the current outputs to the current layers outputs # (1 X nextLayerD)
            if (i != self.layers - 1):
                currentInputs = self.outputsByLayerActivated[self.layers - i - 2] #(1 X prevLayerD)
            else:
                currentInputs = inputs #(1 X inD)
            # If this is the earliest layer the inputs are the inputs, else they are the previous layers outputs
            if (i != 0):
            # The last layer has no outgoing weights and no previous deltas
                outgoingWeights = self.weightsByLayer[self.layers - i] #(nextLayerD X curLayerD)
                protoNewDeltas = np.matmul(delta, outgoingWeights.T) # Delta is outgoing weights times the previous deltas (1 X curLayerD)
                newDeltas = np.multiply(protoNewDeltas, list(map(self.layerAct[self.layers - 1 - i][1], outputs[0]))) #(1 X curLayerD)
            else:
                newDeltas = np.multiply(diffs, list(map(self.layerAct[self.layers - 1 - i][1], outputs[0]))) #(1 X curLayerD)
            # if (len(newDeltas) == 1 and len(newDeltas[0]) == 1):
            # # Basically a test to see if we have a vector or just a single value
            #     change = np.multiply(learnRate, np.multiply(np.transpose(newDeltas), currentInputs)) #(nextD X prevD)
            # else:
                #change = np.atleast_2d(np.multiply(learnRate, np.matmul(np.transpose(newDeltas), np.atleast_2d(currentInputs)))) 
            change = np.multiply(learnRate, np.matmul(currentInputs.T, newDeltas)) #(curLayerD X prevLayerD)
                # Change is the product of deltas and inputs times learning rate
            changeBias = np.multiply(learnRate, newDeltas) # Bias is just product of learn rate and deltas # (1 X curLayerD)
            changeManifold.append(change)
            biasManifold.append(changeBias)
            # add both to the manifold
            delta = newDeltas
            # set up for next layer
        #print(changeManifold)
        for i in range(self.layers):
        # This section just adds the changes to the weight and bias matricies
            self.weightsByLayer[self.layers - i - 1] = np.add(self.weightsByLayer[self.layers - i - 1], changeManifold[i])
            if self.bias:
                self.biasByLayer[self.layers - i - 1] = np.add(self.biasByLayer[self.layers - i - 1], biasManifold[i])