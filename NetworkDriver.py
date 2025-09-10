import math
import random
import matplotlib.pyplot as plt
import time
from nonArrayBased.MyNeuron import neuron as Neuron
from nonArrayBased.MyLayer import layer as Layer
from nonArrayBased.MyNetwork import network as Network
from ArrayBased.ArrayNetwork import ArrayNetworkFeedforward as ArrayNetwork
import numpy as np

@staticmethod
def default(input): # functional
    return input

@staticmethod
def relu(input): # untested
    if input < 0:
        return 0
    return input

@staticmethod
def sigmoid(input): # untested
    return 1.0 / (1 + math.exp(-input))

@staticmethod
def sigmoidDerivative(input): # untested
    return sigmoid(input) * (1 - sigmoid(input))

@staticmethod
def tanh(input): # untested
    return math.tanh(input)

@staticmethod
def softmax(inputs): # untested
    sumE = 0
    for input in inputs:
        sumE += math.exp(input)
    return [input / sumE for input in inputs]

@staticmethod
def softplus(input): # untested
    return math.log(1+math.exp(input))

@staticmethod
def approxDerivative(function, point): # untested
    diff = function(point + .0000001) - function(point)
    return diff / .0000001


if __name__ == "__main__":
    testSet3 = []
    test = [np.atleast_2d([1,1,1,1]), np.atleast_2d([1,1])]
    testSet3.append(test)
    test = [np.atleast_2d([1,1,1,0]), np.atleast_2d([1,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([1,1,0,1]), np.atleast_2d([.5,1])]
    testSet3.append(test)
    test = [np.atleast_2d([1,1,0,0]), np.atleast_2d([.5,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([1,0,1,1]), np.atleast_2d([1,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([1,0,1,0]), np.atleast_2d([1,0])]
    testSet3.append(test)
    test = [np.atleast_2d([1,0,0,1]), np.atleast_2d([.5,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([1,0,0,0]), np.atleast_2d([.5,0])]
    testSet3.append(test)
    test = [np.atleast_2d([0,1,1,1]), np.atleast_2d([.5,1])]
    testSet3.append(test)
    test = [np.atleast_2d([0,1,1,0]), np.atleast_2d([.5,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([0,1,0,1]), np.atleast_2d([0,1])]
    testSet3.append(test)
    test = [np.atleast_2d([0,1,0,0]), np.atleast_2d([0,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([0,0,1,1]), np.atleast_2d([.5,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([0,0,1,0]), np.atleast_2d([.5,0])]
    testSet3.append(test)
    test = [np.atleast_2d([0,0,0,1]), np.atleast_2d([0,.5])]
    testSet3.append(test)
    test = [np.atleast_2d([0,0,0,0]), np.atleast_2d([0,0])]
    testSet3.append(test)

    testSet2 = []
    test = [np.atleast_2d([0,1]), np.atleast_2d([1])]
    testSet2.append(test)
    test = [np.atleast_2d([1,1]), np.atleast_2d([1])]
    testSet2.append(test)
    test = [np.atleast_2d([1,0]), np.atleast_2d([1])]
    testSet2.append(test)
    test = [np.atleast_2d([0,0]), np.atleast_2d([0])]
    testSet2.append(test)

    networkA = Network()
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    A1 = Neuron(weights, 4, random.uniform(-2, 2))
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    B1 = Neuron(weights, 4, random.uniform(-2, 2))
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    C1 = Neuron(weights, 4, random.uniform(-2, 2))
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    D1 = Neuron(weights, 4, random.uniform(-2, 2))
    layerA = Layer([A1, B1, C1, D1])
    layerA.setActivationFunc(sigmoid)
    networkA.addLayer(layerA)
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    A2 = Neuron(weights, 4, random.uniform(-2, 2))
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    B2 = Neuron(weights, 4, random.uniform(-2, 2))
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    C2 = Neuron(weights, 4, random.uniform(-2, 2))
    layerB = Layer([A2, B2, C2])
    layerB.setActivationFunc(sigmoid)
    networkA.addLayer(layerB)
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    A3 = Neuron(weights, 3, random.uniform(-2, 2))
    weights = [random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(-.5, .5)]
    B3 = Neuron(weights, 3, random.uniform(-2, 2))
    # weights = [1, 0, 1]
    # B1 = Neuron(weights, 3, 0)
    layerC = Layer([A3, B3])
    layerC.setActivationFunc(sigmoid)
    networkA.addLayer(layerC)

    num = 000000
    errorTrack = []
    c = 0
    error = 0
    testNum = 0
    #networkA.printInWeights()
    startTime = time.perf_counter()
    for i in range(num):
        error = 0
        testNum = random.randint(0, len(testSet3) - 1)#(testNum + 1) % 4#random.randint(0, 3)
        inputs = testSet3[testNum][0]
        targets = testSet3[testNum][1]
        outputs = networkA.evaluateNetwork(inputs)
        diffs = [targets[i] - outputs[i] for i in range(len(targets))]
        for i in range(len(outputs)):
            error += .5 * (diffs[i]) ** 2
        errorTrack.append(error)
        # print("inputs: ", inputs)
        # print("outputs: ", outputs)
        # print("targets: ", targets)
        # print("diffs: ", diffs)
        # print("Orignal Network: ")
        #networkA.printInWeights()
        networkA.updateNetworkGeneralizedDelta(diffs, .05, sigmoidDerivative)
        #print("new network: ")
        #networkA.printInWeights()
        #print()
    endTime = time.perf_counter()
    print("end, time elapsed: ", endTime-startTime)
    # #networkA.printInWeights()
    # plt.plot(errorTrack)
    # plt.show()
    # inputs = [1,1,1,1]
    # outputs = networkA.evaluateNetwork(inputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)
    # #print("normalized outputs: ", normalOutputs)
    # inputs = [0,1,0,1]
    # outputs = networkA.evaluateNetwork(inputs)
    # #normalOutputs = normalize(outputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)
    # #print("normalized outputs: ", normalOutputs)
    # inputs = [1,0,1,1]
    # outputs = networkA.evaluateNetwork(inputs)
    # #normalOutputs = normalize(outputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)
    # #print("normalized outputs: ", normalOutputs)
    # inputs = [0,0,0,0]
    # outputs = networkA.evaluateNetwork(inputs)
    # #normalOutputs = normalize(outputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)
    # #print("normalized outputs: ", normalOutputs)

    # #networkA.printInWeights()

    #arrayA = np.atleast_2d([1, 2])
    #print(arrayA.T)
    #arrayB = np.atleast_2d([[2,2], [2,2]])
    #print(arrayB)
    #print(np.matmul(arrayB, arrayA.T))



    networkB = ArrayNetwork(4, 2, [2, 2], [[sigmoid, sigmoidDerivative], [sigmoid, sigmoidDerivative]], bias = True, random=True)
    inputs = [2, 2]
    networkB.printNetwork()
    #print("Output:", networkB.evaluateNetwork(inputs))
    #networkB.printNetwork()
    #print("GAP")

    num = 0
    errorTrack = []
    testNum = 0
    #networkA.printInWeights()
    startTime = time.perf_counter()
    learnRate = .25
    for i in range(num):
        error = 0
        testNum = random.randint(0, len(testSet3) - 1)#(testNum + 1) % 4#random.randint(0, 3)
        inputs = testSet3[testNum][0]
        targets = testSet3[testNum][1]
        outputs = networkB.evaluateNetwork(inputs)
        diffs = np.subtract(targets, outputs)
        networkB.updateNetworkGeneralizedDelta(inputs, diffs, learnRate)
        # for i in range(len(testSet3)):
        #     outputs = networkB.evaluateNetwork(testSet3[i][0])
        #     diffs = np.subtract(testSet3[i][1], outputs).tolist()
        #     for i in range(len(outputs)):
        #         error += .5 * (diffs[0][i]) ** 2
        for i in range(len(outputs)):
                error += .5 * (diffs[0][i]) ** 2
        errorTrack.append(error)
        learnRate *= 1/((1+learnRate*i))

    endTime = time.perf_counter()
    print("end, time elapsed: ", endTime-startTime)
    # plt.plot(errorTrack)
    # plt.show()
    # networkB.printNetwork()
    # inputs = testSet3[1][0]
    # outputs = networkB.evaluateNetwork(inputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)
    # inputs = testSet3[0][0]
    # outputs = networkB.evaluateNetwork(inputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)
    # inputs = testSet3[2][0]
    # outputs = networkB.evaluateNetwork(inputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)
    # inputs = testSet3[3][0]
    # outputs = networkB.evaluateNetwork(inputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)
    # inputs = testSet3[10][0]
    # outputs = networkB.evaluateNetwork(inputs)
    # print("inputs: ", inputs)
    # print("outputs: ", outputs)

    for bptt_step in np.arange(max(0, 2), 7)[::-1]:
        print(bptt_step)

