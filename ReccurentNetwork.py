from scipy.special import softmax

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords 
import nltk
#nltk.download('punkt_tab')
#nltk.download('stopwords')
import nltk
import string

import time

from Misc.Loader import loadMatrix, loadWords

@staticmethod
def sigmoid(input): # untested
    #print(input)
    return 1.0 / (1 + np.exp(-input))

@staticmethod
def sigmoidDerivative(input): # untested
    return sigmoid(input) * (1 - sigmoid(input))

class RecurrentNetwork:
    hiddenWeights = []
    embeddingWeights = []
    outputWeights = []
    embeddingMatrix = []
    hiddenState = []
    oneHotInput = []
    wordIndex = []
    wordCount = 0
    words = []
    trainingData = []

    def __init__(self, embeddingDim, hiddenDim, wordCount, wordIndex, embeddingMatrix, words):
        self.hiddenWeights = np.random.uniform(-.8, .8, (hiddenDim, hiddenDim))
        self.embeddingWeights = np.random.uniform(-.8, .8, (hiddenDim, embeddingDim))
        self.outputWeights = np.random.uniform(-.8, .8, (wordCount, hiddenDim))
        self.hiddenState = np.random.uniform(-.8, .8, (hiddenDim, 1))
        self.defaultHiddenState = self.hiddenState
        self.wordIndex = wordIndex
        self.wordCount = wordCount
        self.embeddingMatrix = embeddingMatrix
        self.words = words

    def predictNext(self, inputString):
        splitString = inputString.split()
        stop_words = set(stopwords.words('english'))
        x = [word.strip(string.punctuation) for word in splitString if word not in stop_words] # Removes punctuation words
        x = [word.lower() for word in x] # to lowercase

        output = self.passThrough(x)

        wordNProb = []
        for i in range(self.wordCount):
            wordNProb.append([output[i][0], self.words[i]])
            #print(type(output[i][0]))
        index = np.argmax(wordNProb, 0)
        #print(index)
        newOutput = output.flatten().tolist()
        #print(wordNProb)
        #print(np.random.choice(self.words, p=newOutput))
        return np.random.choice(self.words, p=newOutput)

    def passThrough(self, tokens):
        self.hiddenStateRecord = []
        for token in tokens:
            self.updateHidden(token)

        output = softmax(np.dot(self.outputWeights, self.hiddenState))
        return output
    
    def preprocess(self):
        with open("inputCorpus.txt", 'r', encoding='utf-8') as file:
            corpus = file.read()
        sentences = nltk.PunktSentenceTokenizer().tokenize(corpus)
        tokenizedSentences = []
        for sentence in sentences:
            tokenizedSentence = []
            for word in nltk.word_tokenize(sentence):
                if (word not in stopwords.words()) and (word not in string.punctuation):
                    tokenizedSentence.append(word.lower())
            tokenizedSentences.append(tokenizedSentence)
        self.trainingData = tokenizedSentences

    def train(self, epochs):
        learning_rate = .005
        lossTrack = []
        for i in range(epochs):
            if (i % 10 == 0):
                loss = self.calculateLoss()
                print("Epoch: ", i, " Loss: ", loss)
                lossTrack.append(loss)
                # adjust the learning rate if loss increases
                if (len(lossTrack) > 1 and lossTrack[-1] > lossTrack[-2]):
                    learning_rate = learning_rate * 0.5
                    print("setting learning rate to %f" %(learning_rate))
            # for each training example...
            for sentence in self.trainingData:
                length = len(sentence)
                rawOutput = self.passThrough(sentence)
                output = softmax(rawOutput)
                totalChangeHidden = np.zeros(self.hiddenWeights.shape)
                totalChangeOutout = np.zeros(self.outputWeights.shape)
                totalChangeInput = np.zeros(self.embeddingWeights.shape)
                for j in reversed(range(length - 1)):
                    target = np.zeros((self.wordCount, 1))
                    target[self.wordIndex[sentence[j+1]]] = 1

                    diffs = np.subtract(target, output)
                    changeOutput, changeHidden, changeInput = self.backpropagate(diffs, sentence, j+1)
                    totalChangeOutout += changeOutput
                    totalChangeHidden += changeHidden
                    totalChangeInput += changeInput
                if np.linalg.norm(totalChangeHidden, ord=2) > 3:
                    totalChangeHidden = (3.0 / np.linalg.norm(totalChangeHidden, ord=2)) * totalChangeHidden
                if np.linalg.norm(totalChangeOutout, ord=2) > 3:
                    totalChangeOutout = (3.0 / np.linalg.norm(totalChangeOutout, ord=2)) * totalChangeOutout
                if np.linalg.norm(totalChangeInput, ord=2) > 3:
                    totalChangeInput = (3.0 / np.linalg.norm(totalChangeInput, ord=2)) * totalChangeInput
                self.hiddenWeights += learning_rate * totalChangeHidden
                self.outputWeights += learning_rate * totalChangeOutout
                self.embeddingWeights += learning_rate * totalChangeInput
        plt.plot(lossTrack)
        plt.show()

    def calculateTotalLoss(self):
        L = 0
        # for each sentence ...
        for i in range(len(self.trainingData)):
            self.hiddenStateRecord = []
            outputRecord = []
            for token in self.trainingData[i]:
                self.updateHidden(token)
                outputRecord.append(softmax(np.dot(self.outputWeights, self.hiddenState)))
            # we only care about our prediction of the "correct" words
            correctWordPredictions = []
            for j in range(len(self.trainingData[i]) - 1):
                correctWordPredictions.append(outputRecord[j][self.wordIndex[self.trainingData[i][j+1]]])

            # add to the loss based on how off we were
            L += -1 * np.sum(np.log(correctWordPredictions))
        return L

    def calculateLoss(self):
        # divide the total loss by the number of training examples
        N = np.sum((len(sentence) for sentence in self.trainingData))
        return self.calculateTotalLoss() / N
    
    def backpropagate(self, diffs, sentence, lengthOneMore):
        partialChangeInOutputWeight = np.dot(diffs, self.hiddenState.T)
        partialChangeInHiddenWeight = np.zeros(self.hiddenWeights.shape)
        partialChangeInInputWeight = np.zeros(self.embeddingWeights.shape)
        d = np.dot(self.outputWeights.T, diffs)
        d = d * (1 - self.hiddenStateRecord[lengthOneMore - 1] ** 2)
        for k in reversed(range(lengthOneMore)):
            inputEncoded = np.zeros((self.wordCount, 1))
            inputEncoded[self.wordIndex[sentence[lengthOneMore]]] = 1
            embeddedVector = np.dot(self.embeddingMatrix, inputEncoded)
            partialChangeInInputWeight += np.outer(d, embeddedVector)
            #partialChangeInInputWeight[:, self.wordIndex[sentence[k]]] += np.multiply(d, embeddedVector) # Doesn't work as is, I think they made an assumption I cannot
            if (k != 0):
                partialChangeInHiddenWeight += np.outer(d, self.hiddenStateRecord[k-1])
                d = np.dot(self.hiddenWeights.T, d) * (1 - self.hiddenStateRecord[k-1] ** 2)
            else:
                partialChangeInHiddenWeight += np.outer(d, self.defaultHiddenState)
                #d = np.dot(self.hiddenWeights.T, d) * (1 - self.defaultHiddenState ** 2)
        return partialChangeInOutputWeight, partialChangeInHiddenWeight, partialChangeInInputWeight

    def updateHidden(self, token):
        encoding = np.zeros((self.wordCount, 1))
        encoding[self.wordIndex[token]] = 1
        embeddedVector = np.dot(self.embeddingMatrix, encoding)
        self.hiddenState = np.tanh(np.atleast_2d(list(map(sigmoid ,np.dot(self.hiddenWeights, self.hiddenState) + np.dot(self.embeddingWeights, embeddedVector)))).T).T
        #TODO we have some kind of exploding size of the hidden state as the epochs go, not sure why
        self.hiddenStateRecord.append(self.hiddenState)
    
    def writeWeights(self):
        with open("recurrentHiddenWeights.txt", "w") as txt_file:
            for line in self.hiddenWeights:
                txt_file.write(" ".join(map(str, line)) + "\n")
            txt_file.close()
        with open("recurrentEmbeddingWeights.txt", "w") as txt_file:
            for line in self.embeddingWeights:
                txt_file.write(" ".join(map(str, line)) + "\n")
            txt_file.close()
        with open("recurrentOutputWeights.txt", "w") as txt_file:
            for line in self.outputWeights:
                txt_file.write(" ".join(map(str, line)) + "\n")
            txt_file.close()

    def readWeights(self):
        self.hiddenWeights = loadMatrix("recurrentHiddenWeights.txt")
        self.embeddingWeights = loadMatrix("recurrentEmbeddingWeights.txt")
        self.outputWeights = loadMatrix("recurrentOutputWeights.txt")

if __name__ == '__main__':
    embeddings = loadMatrix('wordEmbedding.txt')
    matrix = np.atleast_2d(embeddings).T
    words = loadWords('vocab.txt')
    word2Index = {}
    for i in range(len(words)):
        word2Index[words[i].lower()] = i
    embeddingSize = len(embeddings[0])
    net = RecurrentNetwork(embeddingSize, 30, len(words), word2Index, matrix, words)
    net.preprocess()
    if True:
        net.train(100)
    else:
        net.readWeights()
    net.writeWeights()
    generateSentenceLength = 10
    sentence = "cookies"
    i = 1
    while i <= generateSentenceLength:
        next = net.predictNext(sentence)
        sentence += " " + next
        i += 1
        print(sentence)


