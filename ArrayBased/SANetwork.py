import numpy as np
import random as rand
from typing import List
import math
import matplotlib.pyplot as plt
import time

class ArrayNetworkFeedforward:
    """ 
    Class for a neural network implemented using arrays using simulated annealing
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
    updateNetworkSimulatedAnnealing(self, inputs, targets, temperature)
        Updates the network weights using simulated annealing
    calculateError(self, inputs, targets)
        Calculates the error for given inputs and targets
    perturbNetwork(self, perturbationSize)
        Creates a perturbed copy of the network
    copyWeightsAndBiases(self)
        Returns a copy of current weights and biases
    restoreWeightsAndBiases(self, savedState)
        Restores weights and biases from saved state
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
        self.weightsByLayer = []
        self.biasByLayer = []
        self.layerLengths = []
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
                bias_arr = []
                for j in range(neuronPerLayer[i]):
                    if (random):
                        bias_arr.append(rand.uniform(-.8,.8))
                    else:
                        bias_arr.append(0)
                self.biasByLayer.append(np.atleast_2d(bias_arr))
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

    def calculateError(self, testSet):
        """ Calculates the error for given inputs and targets
         
        Parameters:
        -----------
            inputs : double array
                Vector of input values, (1XD)
            targets : double array
                Vector of target output values, (1XD)
                
        Returns:
        --------
            error : double
                Sum of squared errors
        """
        error = 0.0
        for test in testSet:
            inputs = test[0]
            targets = test[1]
            outputs = self.evaluateNetwork(inputs)
            diffs = np.subtract(targets, outputs)
            for i in range(len(outputs[0])):
                error += 0.5 * (diffs[0][i]) ** 2
        return error/len(testSet)
    
    def showCaseNetwork(self, testSet):
        """ Calculates the error for given inputs and targets
         
        Parameters:
        -----------
            inputs : double array
                Vector of input values, (1XD)
            targets : double array
                Vector of target output values, (1XD)
                
        Returns:
        --------
            error : double
                Sum of squared errors
        """
        for test in testSet:
            inputs = test[0]
            targets = test[1]
            outputs = self.evaluateNetwork(inputs)
            diffs = np.subtract(targets, outputs)
            error = 0.0
            for i in range(len(outputs[0])):
                error += 0.5 * (diffs[0][i]) ** 2
            print(f"Test {i}: Input {inputs}, Target {targets}, Output {outputs}, Error: {error:.4f}")

    def copyWeightsAndBiases(self):
        """ Returns a copy of current weights and biases
        
        Returns:
        --------
            tuple : (list of weight arrays, list of bias arrays)
        """
        weightsCopy = [w.copy() if isinstance(w, np.ndarray) else w for w in self.weightsByLayer]
        biasesCopy = [b.copy() if isinstance(b, np.ndarray) else b for b in self.biasByLayer]
        return (weightsCopy, biasesCopy)

    def restoreWeightsAndBiases(self, savedState):
        """ Restores weights and biases from saved state
         
        Parameters:
        -----------
            savedState : tuple
                Tuple containing (weights, biases) to restore
        """
        weights, biases = savedState
        self.weightsByLayer = [w.copy() if isinstance(w, np.ndarray) else w for w in weights]
        self.biasByLayer = [b.copy() if isinstance(b, np.ndarray) else b for b in biases]

    def perturbNetwork(self, perturbationSize):
        """ Creates a perturbed version of the current network
         
        Parameters:
        -----------
            perturbationSize : double
                Maximum amount to perturb each weight/bias
        """
        for i in range(self.layers):
            # Perturb weights
            perturbation = np.random.uniform(-perturbationSize, perturbationSize, 
                                            self.weightsByLayer[i].shape)
            self.weightsByLayer[i] = np.add(self.weightsByLayer[i], perturbation)
            
            # Perturb biases
            if self.bias:
                biasPerturbation = np.random.uniform(-perturbationSize, perturbationSize, 
                                                     self.biasByLayer[i].shape)
                self.biasByLayer[i] = np.add(self.biasByLayer[i], biasPerturbation)

    def updateNetworkSimulatedAnnealing(self, testSet, temperature, perturbationSize=0.1, batchSize=8):
        """ Updates the network weights using simulated annealing
         
        Parameters:
        -----------
            testSet : list of tuples
                List of (input, target) pairs for training
            temperature : double
                Current temperature for annealing process
            perturbationSize : double
                Size of random perturbations to weights
            batchSize : int
                Number of perturbations to test per update
                
        Returns:
        --------
            error : double
                New error after update
        """
        # Calculate current error
        currentError = self.calculateError(testSet)
        
        # Save current state
        originalState = self.copyWeightsAndBiases()
        errors_states = []

        # Perturb the network
        self.perturbNetwork(perturbationSize)
        
        # Calculate new error
        newError = self.calculateError(testSet)
        
        # Calculate change in error
        deltaError = newError - currentError
        errors_states.append([deltaError, self.copyWeightsAndBiases()])

        for i in range(batchSize - 1):
            self.restoreWeightsAndBiases(originalState)
            # Perturb the network
            self.perturbNetwork(perturbationSize)
            # Calculate new error
            newError = self.calculateError(testSet)
            
            # Calculate change in error
            deltaError = newError - currentError
            errors_states.append([deltaError, self.copyWeightsAndBiases()])
        # Find the best perturbation in the batch
        errors_states.sort(key=lambda x: x[0])
        if errors_states[0][0] < 0:
            self.restoreWeightsAndBiases(errors_states[0][1])
            newError = errors_states[0][0]
            return True, newError
        else:
            acceptanceProbability = list(map(lambda x: math.exp(-x[0] / temperature), errors_states))
            totalProbability = sum(acceptanceProbability)
            tempComp = min(1/temperature, 1e-3)
            if totalProbability < tempComp:
                self.restoreWeightsAndBiases(originalState)
                return False, currentError
            acceptanceProbability = list(map(lambda x: x / totalProbability, acceptanceProbability))
            chosenStateIndex = np.random.choice(range(len(errors_states)), p=acceptanceProbability)
            self.restoreWeightsAndBiases(errors_states[chosenStateIndex][1])
            newError = errors_states[chosenStateIndex][0]
            return True, newError
            # Accept worse solution with probability based on temperature

@staticmethod
def sigmoid(input): # untested
    return 1.0 / (1 + math.exp(-input))

@staticmethod
def sigmoidDerivative(input): # untested
    return sigmoid(input) * (1 - sigmoid(input))

if __name__ == "__main__":
    testSet2 = []
    test = [np.atleast_2d([0,1]), np.atleast_2d([1])]
    testSet2.append(test)
    test = [np.atleast_2d([1,1]), np.atleast_2d([0])]
    testSet2.append(test)
    test = [np.atleast_2d([1,0]), np.atleast_2d([1])]
    testSet2.append(test)
    test = [np.atleast_2d([0,0]), np.atleast_2d([0])]
    testSet2.append(test)

    testSet3 = []
    test = [np.atleast_2d([1,1,1,1]), np.atleast_2d([1,1])]
    testSet3.append(test)
    test = [np.atleast_2d([1,1,1,0]), np.atleast_2d([1,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([1,1,0,1]), np.atleast_2d([1,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([1,1,0,0]), np.atleast_2d([1,0])]
    testSet3.append(test)
    test = [np.atleast_2d([1,0,1,1]), np.atleast_2d([.5,1])]
    testSet3.append(test)
    test = [np.atleast_2d([1,0,1,0]), np.atleast_2d([.5,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([1,0,0,1]), np.atleast_2d([.5,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([1,0,0,0]), np.atleast_2d([.5,0])]
    testSet3.append(test)
    test = [np.atleast_2d([0,1,1,1]), np.atleast_2d([.5,1])]
    testSet3.append(test)
    test = [np.atleast_2d([0,1,1,0]), np.atleast_2d([.5,.5])]
    testSet3.append(test)
    # test = [np.atleast_2d([0,1,0,1]), np.atleast_2d([.5,.5])]
    # testSet3.append(test)
    test = [np.atleast_2d([0,1,0,0]), np.atleast_2d([.5,0])]
    testSet3.append(test)
    test = [np.atleast_2d([0,0,1,1]), np.atleast_2d([0, 1])]
    testSet3.append(test)
    test = [np.atleast_2d([0,0,1,0]), np.atleast_2d([0,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([0,0,0,1]), np.atleast_2d([0,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([0,0,0,0]), np.atleast_2d([0,0])]
    testSet3.append(test)
    
    networkB = ArrayNetworkFeedforward(4, 2, [2, 2], [[sigmoid, sigmoidDerivative], [sigmoid, sigmoidDerivative]], bias = True, random=True)
    networkB.printNetwork()

    num_epochs = 1000
    iterations_per_epoch = 1000//len(testSet3)
    errorTrack = []
    acceptanceTrack = []
    epochErrorTrack = []
    testNum = 0
    startTime = time.perf_counter()
    
    # Simulated annealing parameters
    initialTemp = 5
    finalTemp = 0.01
    coolingRate = 0.9995  # Temperature multiplier per iteration
    perturbationSize = .1
    
    temperature = initialTemp
    temp_track = [initialTemp]
    
    for epoch in range(num_epochs):
        epoch_errors = []
        epoch_acceptances = []
        
        for iteration in range(iterations_per_epoch):
            accepted, error = networkB.updateNetworkSimulatedAnnealing(testSet3, temperature, perturbationSize)
            
            errorTrack.append(error)
            acceptanceTrack.append(1 if accepted else 0)
            epoch_errors.append(error)
            epoch_acceptances.append(1 if accepted else 0)
            
            # Cool down temperature
            temperature = max(finalTemp, temperature * coolingRate)
            temp_track.append(temperature)
        
        # Calculate epoch statistics
        avg_epoch_error = sum(epoch_errors) / len(epoch_errors)
        acceptance_rate = sum(epoch_acceptances) / len(epoch_acceptances)
        epochErrorTrack.append(avg_epoch_error)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Temperature: {temperature:.4f}, "
                  f"Avg Error: {avg_epoch_error:.4f}, Acceptance Rate: {acceptance_rate:.3f}")
        
        # Early stopping if temperature is too low
        if temperature <= finalTemp:
            print(f"Temperature reached minimum at epoch {epoch}")
            break

    endTime = time.perf_counter()
    print("end, time elapsed: ", endTime-startTime)
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: All errors
    ax1.plot(errorTrack)
    ax1.set_ylabel('Error')
    ax1.set_xlabel('Iteration')
    ax1.set_title('Error over all iterations')
    ax1.grid(True)
    
    # Plot 2: Epoch average errors
    ax2.plot(epochErrorTrack)
    ax2.set_ylabel('Average Error')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Average Error per Epoch')
    ax2.grid(True)
    # Plot 3: Log scale of errors
    window = 1000
    acceptance_avg = [sum(acceptanceTrack[max(0,i-window):i+1])/min(window, i+1) for i in range(len(acceptanceTrack))]
    ax3.plot(acceptance_avg)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Acceptance Rate')
    ax3.set_title('Moving Average Acceptance Rate')
    ax3.grid(True)
    
    # Plot 4: Temperature over iterations
    # temp_track = [initialTemp * (coolingRate ** i) for i in range(len(errorTrack))]
    ax4.plot(temp_track)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Temperature Schedule')
    ax4.set_yscale('log')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Test the network on all test cases
    test = [np.atleast_2d([0,1,0,1]), np.atleast_2d([.5,.5])]
    testSet3.append(test)
    print("\nFinal Network Performance:")
    error = networkB.calculateError(testSet3)
    networkB.showCaseNetwork(testSet3)
    print(f"Average Test Error: {error:.4f}")