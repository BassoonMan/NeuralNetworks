import math
import random
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from nonArrayBased.MyNeuron import neuron as Neuron
from nonArrayBased.MyLayer import layer as Layer
from nonArrayBased.MyNetwork import network as Network
from ArrayBased.ArrayNetwork import ArrayNetworkFeedforward as ArrayNetwork
import numpy as np

from nltk.corpus import stopwords 
import string

from scipy.special import softmax

def linear(x):
    return x
def linDerivative(x):
    return 1

class Word2Vec:
    corpus = [] # Tokenized corpus
    words = [] # Array of every unique word in alphebetical order
    net = None # neural network that achieves embeddings
    uniqueWordNum = 0
    hiddenLayerDim = 0
    windowSize = 0
    xTrain = []
    yTrain = []
    wordIndex = {}
    learnRate = .0005
    def __init__(self, hiddenLayerDim, windowSize):
        self.hiddenLayerDim = hiddenLayerDim
        self.windowSize = windowSize
        with open("inputCorpus.txt", 'r', encoding='utf-8') as file:
            self.corpus = self.preprocess(file.read())
        data = {}

        for sentence in self.corpus:
            for word in sentence:
                if word not in data:
                    data[word] = 1
                else:
                    data[word] += 1
        # This section counts frequency of words inside the corpus

        self.uniqueWordNum = len(data)

        data = sorted(list(data.keys())) # Sort the words in alphebetical order I think
        self.words = data 

        vocab = {}
        for i in range(len(data)):
            vocab[data[i]] = i
        self.wordIndex = vocab
        # Establish conversion from word to index, ie using word can get index
        
        for sentence in self.corpus:
            for i in range(len(sentence)): # Each iteratation slides the window down one
                centerWord = [0 for x in range(self.uniqueWordNum)] # Create 0 vector with the vocab list as its length
                centerWord[vocab[sentence[i]]] = 1 # marks the current word's index as 1
                context = [0 for x in range(self.uniqueWordNum)] # Create similar 0 vector

                for j in range(i - self.windowSize, i + self.windowSize): # This loop goes over the entire window
                    if i != j and j >= 0 and j < len(sentence): # eliminates impossible words and center word
                        context[vocab[sentence[j]]] += 1 # counts number of occurence of each context word within the current window
                        self.xTrain.append(centerWord)
                        self.yTrain.append(context)
        #print(self.corpus)
        self.net = ArrayNetwork(self.uniqueWordNum, 2, [self.hiddenLayerDim, self.uniqueWordNum], [[linear, linDerivative], [linear, linDerivative]], bias = False, random=True)
        #self.net.printNetwork()

    def preprocess(self, corpus):
        stop_words = set(stopwords.words('english'))
        training_data = []
        sentences = corpus.split(".")
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip() # Not sure purpose
            sentence = sentences[i].split() # Takes a sentance and splits it into individual words
            x = [word.strip(string.punctuation) for word in sentence if word not in stop_words] # Removes punctuation words
            x = [word.lower() for word in x] # to lowercase
            training_data.append(x)
        return training_data
    
    def train(self, epochs):
        errorTrack = []
        startTime = time.perf_counter()
        learnRate = .001
        for i in range(1, epochs):
            loss = 0
            for j in range(len(self.xTrain)):
                inputs = np.atleast_2d(self.xTrain[j])
                targets = np.atleast_2d(self.yTrain[j])
                rawOutputs = self.net.evaluateNetwork(inputs)
                outputs = softmax(rawOutputs)
                #print(outputs)
                diffs = np.subtract(targets, outputs)
                self.net.updateNetworkGeneralizedDelta(inputs, diffs, learnRate)
                C = 0
                #Loss is 

                # for m in range(self.uniqueWordNum):
                #     if(self.yTrain[j][m]):
                #         loss += -1*rawOutputs[0][m]
                #         C += 1
                # loss += C*np.log(np.sum(np.exp(rawOutputs)))
            print("epoch ", i, " loss = ", loss)
            errorTrack.append(loss)
            learnRate *= 1/((1+learnRate*i))
            # wordEmbedding = self.net.outputEmbedding()
            # labels = np.array(list(self.wordIndex.keys()))
            # wordX = np.transpose(wordEmbedding)[0]
            # wordY = np.transpose(wordEmbedding)[1]
            # scatter = plt.scatter(wordX, wordY)
            # for i, txt in enumerate(labels):
            #     plt.annotate(txt, (wordX[i], wordY[i]), textcoords="offset points", xytext=(0,10), ha='center')
            # plt.show()
            # scatter.remove()

        endTime = time.perf_counter()
        print("end, time elapsed: ", endTime-startTime)
        plt.plot(errorTrack)
        plt.show()

    def animateEmbeddings(self, epochs):
        fig, ax = plt.subplots(figsize=(10,8))
        ax.autoscale()
        wordEmbedding = self.net.outputEmbedding()
        wordX = np.transpose(wordEmbedding)[0]
        wordY = np.transpose(wordEmbedding)[1]
        labels = np.array(list(self.wordIndex.keys()))
        scatter = ax.scatter(wordX, wordY)
        annotations = []
        for i, txt in enumerate(labels):
            annotations.append(ax.annotate(txt, (wordX[i], wordY[i]), textcoords="offset points", xytext=(0,10), ha='center'))
        errorTrack = []
        def animate(i):
            loss = 0
            for j in range(len(self.xTrain)):
                inputs = np.atleast_2d(self.xTrain[j])
                targets = np.atleast_2d(self.yTrain[j])
                rawOutputs = self.net.evaluateNetwork(inputs)
                outputs = softmax(rawOutputs)
                #print(outputs)
                diffs = np.subtract(targets, outputs)
                self.net.updateNetworkGeneralizedDelta(inputs, diffs, self.learnRate)
                C = 0
                for m in range(self.uniqueWordNum):
                    if(self.yTrain[j][m]):
                        loss += -1*rawOutputs[0][m]
                        C += 1
                loss += C*np.log(np.sum(np.exp(rawOutputs)))
            print("epoch ", i, " loss = ", loss)
            self.learnRate *= 1/((1+self.learnRate*(i + 1)))
            wordEmbedding = self.net.outputEmbedding()
            labels = np.array(list(self.wordIndex.keys()))
            wordX = np.transpose(wordEmbedding)[0]
            wordY = np.transpose(wordEmbedding)[1]
            # for i, txt in enumerate(labels):
            #     ax.annotate(txt, (wordX[i], wordY[i]), textcoords="offset points", xytext=(0,10), ha='center')
            # plt.show()
            combined = np.vstack((wordX, wordY))
            xpos = np.max(wordX)
            xneg = np.min(wordX)
            ypos = np.max(wordY)
            yneg = np.min(wordY)
            scatter.set_offsets(combined.T)
            for i, txt in enumerate(labels):
                annotations[i].xy = (wordX[i], wordY[i])
            errorTrack.append(loss)
            ax.set_xlim([xneg, xpos])
            ax.set_ylim([yneg, ypos])
            #print([-xneg, xpos])
            #print([-yneg, ypos])

            return (scatter)

        Nstep = epochs
        anim = animation.FuncAnimation(fig=fig, func=animate, 
                                    frames = Nstep, interval=1, blit=False, repeat=False)
        plt.show()
        plt.plot(errorTrack)
        plt.show()

    def writeEmbedding(self):
        self.net.outputEmbedding()
        with open("wordEmbedding.txt", "a") as txt_file:
            txt_file.write('[')
            for line in self.wordIndex:
                txt_file.write('\'' + line + "\',\n") # works with any number of elements in a line
            txt_file.write(']')
            txt_file.close()


    def predict(self, word, number_of_predictions):
        #print(self.wordIndex)
        #self.net.printNetwork()
        if word in self.words:
            index = self.wordIndex[word]
            X = [0 for i in range(self.uniqueWordNum)]
            X[index] = 1
            prediction = softmax(self.net.evaluateNetwork(np.atleast_2d(X)).tolist())
            #print(prediction)
            output = {}
            for i in range(self.uniqueWordNum):
                output[prediction[0][i]] = i
            #print(output)
            top_context_words = []
            for k in sorted(output, reverse=True):
                top_context_words.append(self.words[output[k]])
                if len(top_context_words) >= number_of_predictions:
                    break
            #print(self.wordIndex)
            #self.net.printNetwork()
            return top_context_words
        else:
            print("Word not found in dictionary")

    
if __name__ == "__main__":
    model = Word2Vec(30, 10)
    #model.train(1000)
    #model.animateEmbeddings(1000)
    #model.writeEmbedding()
    print(model.predict("trump", 2))
