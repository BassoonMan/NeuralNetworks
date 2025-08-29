from ArrayBased.ArrayNetwork import ArrayNetworkFeedforward as ArrayNetwork
import numpy as np

from nltk.corpus import stopwords 
import nltk
import string

from scipy.special import softmax


def linear(x):
    return x
def linDerivative(x):
    return 1

class Word2Vec2:
    corpus = [] # Processed Input data, should be the text broken into sentances, which are then broken into tokens array of array style
    contextForCenter = [] # should be encoded vectors for each center word, the target outputs
    centerEncoding = [] # Encoded center vectors, just basically indexed vocab words
    word2Index = {} # Dictionary for taking the vocab words and retrieving their context
    net = [] # neural network trained to achieve outputs
    hiddenDimension = 0 # Dimension of hidden layer, also the dimesnion of the encoding
    windowSize = 0 # length of window about center words to gather context
    wordCount = 0 # Number of unique words

    def __init__(self, hiddenDim, windowSize): # Initialize neural network
        self.net = ArrayNetwork(len(self.centerEncoding), 2, [self.hiddenDimension, len(self.centerEncoding)], [[linear, linDerivative], [linear, linDerivative]], random=True, bias=False)

    def predict(self, word, predictionCount): # Run a center word and give most likely context
        pass

    def writeEmbeddings(self): # Output embedding vectors into file so I can use them elsewhere
        pass

    def preprocess(self): # Tokenize and filter input, then create context and center encodings for network training
        with open("inputCorpus.txt", 'r', encoding='utf-8') as file:
            corpus = file.read()
        sentences = nltk.PunktSentenceTokenizer().tokenize(corpus)
        #print(sentences)
        tokenizedSentences = []
        for sentence in sentences:
            tokenizedSentence = []
            #print(nltk.word_tokenize(sentence))
            for word in nltk.word_tokenize(sentence):
                if (word not in stopwords.words()) and (word not in string.punctuation):
                    tokenizedSentence.append(word.lower())
            tokenizedSentences.append(tokenizedSentence)
        self.corpus = tokenizedSentences
        # Above Tokenizes the data into sentences and tokens

        words = {}
        for sentence in self.corpus:
            for token in sentence:
                if token not in word:
                    words[token] = 1
                else:
                    words[token] += 1
        #print(word)
        words = sorted(list(words.keys()))
        # Counts word frequency and wordCount in dictionary, then sorts by alphebetical order
        # I'm not sure why frequency is included, I may come back and remove that section

        self.wordCount = len(words)

        for i in range(len(words)):
            self.word2Index[words[i]] = i
        #print(self.word2Index)
        # Now given a word I can get its index for 1 hot encoding.

        # Add section to encode center and word stuff.
        for sentence in self.corpus:
            for i in range(self.wordCount):
                center = np.zeros((self.wordCount,))
                center[self.word2Index[sentence[i]]] = 1
                self.centerEncoding.append(center)
                context = np.zeros((self.wordCount,))
                for j in range(i - self.windowSize, i + self.windowSize):
                    if (j >= 0) and (j < len(sentence)) and (j != i):
                        context[sentence[j]] += 1
        # This is as far as I can go from inherent understanding, check above with paper and 1-hot encoding and finish
        



        

    def train(self, epochs):
        pass

if __name__ == "__main__":
    model = Word2Vec2(30, 10)
    model.preprocess()
    model.train(10)
    #model.writeEmbedding()
    print(model.predict("trump", 2))
