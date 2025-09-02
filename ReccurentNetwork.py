from scipy.special import softmax


import math
import numpy as np
from nltk.corpus import stopwords 
import nltk
#nltk.download('punkt_tab')
#nltk.download('stopwords')
import nltk
import string

import time

@staticmethod
def sigmoid(input): # untested
    return 1.0 / (1 + math.exp(-input))

@staticmethod
def sigmoidDerivative(input): # untested
    return sigmoid(input) * (1 - sigmoid(input))

class RecurrentNetwork:
    hiddenWeights = []
    embeddingWeights = []
    outPutWeights = []
    embeddingMatrix = []
    hiddenState = []
    oneHotInput = []
    wordIndex = []
    wordCount = 0
    words = []

    def __init__(self, embeddingDim, hiddenDim, wordCount, wordIndex, embeddingMatrix, words):
        self.hiddenWeights = np.random.rand(hiddenDim, hiddenDim)
        self.embeddingWeights = np.random.rand(hiddenDim, embeddingDim)
        self.outPutWeights = np.random.rand(wordCount, hiddenDim)
        self.hiddenState = np.random.rand(hiddenDim, 1)
        self.wordIndex = wordIndex
        self.wordCount = wordCount
        self.embeddingMatrix = embeddingMatrix
        self.words = words

    def predictNext(self, inputString):
        tokens = []
        for word in nltk.word_tokenize(inputString):
            if (word not in stopwords.words()) and (word not in string.punctuation):
                tokens.append(word.lower())

        output = self.passThrough(tokens)
        wordNProb = []
        for i in range(self.wordCount):
            wordNProb.append([output[i][0], self.words[i]])
        wordNProb = np.sort(wordNProb, axis=0)[::-1]
        print(wordNProb)

    def passThrough(self, tokens):
        for token in tokens:
            encoding = np.zeros((self.wordCount, 1))
            encoding[self.wordIndex[token]] = 1
            embeddedVector = np.dot(self.embeddingMatrix, encoding)
            self.hiddenState = np.dot(self.hiddenWeights, self.hiddenState) + np.dot(self.embeddingWeights, embeddedVector)

        output = softmax(np.dot(self.outPutWeights, self.hiddenState))
        return output

    def train(self, epochs):
        with open("inputCorpusForRNN.txt", 'r', encoding='utf-8') as file:
            corpus = file.read()
        sentences = nltk.PunktSentenceTokenizer().tokenize(corpus)
        tokenizedSentences = []
        for sentence in sentences:
            tokenizedSentence = []
            #print(nltk.word_tokenize(sentence))
            for word in nltk.word_tokenize(sentence):
                if (word not in stopwords.words()) and (word not in string.punctuation):
                    tokenizedSentence.append(word.lower())
            tokenizedSentences.append(tokenizedSentence)

        for i in range(epochs):
            for sentence in sentences:
                curSentence = []
                length = len(sentence)
                for j in range(len(length)):
                    curSentence.append(sentence[j])
                    output = self.passThrough(sentence)
                    #Implement Backpropagation through time
        pass


if __name__ == '__main__':
    matrix = np.atleast_2d([[-1.6497277850278043, 0.5311218701267376],
        [-2.0564834761733453, -0.361953815172465],
        [0.04391998039547603, 0.5880408500070287],
        [-1.8855573035541664, -0.630828802545307],
        [1.3365835877466512, 0.6402821875267963],
        [0.2907640005459061, -0.9067710115915298]]).T
    word2Index = {'hot':0,
                'joshua':1,
                'short':2,
                'tall':3,
                'trump':4,
                'ugly':5,
    }   
    words = ['hot',
            'joshua',
            'short',
            'tall',
            'trump',
            'ugly',
            ]
    net = RecurrentNetwork(2, 2, 6, word2Index, matrix, words)
    net.predictNext("Trump is")


