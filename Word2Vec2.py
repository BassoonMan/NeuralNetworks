from ArrayBased.ArrayNetwork import ArrayNetworkFeedforward as ArrayNetwork
import numpy as np

from nltk.corpus import stopwords 
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import nltk
import string

from scipy.special import softmax


def linear(x):
    return x
def linDerivative(x):
    return 1

class Word2Vec2:
    corpus = [] # Processed Input data, should be the text broken into sentances, which are then broken into tokens array of array style
    contextWords = [] # should be encoded vectors for each center word, the target outputs
    centerWords = [] # Encoded center vectors, just basically indexed vocab words
    word2Index = {} # Dictionary for taking the vocab words and retrieving their context
    net = [] # neural network trained to achieve outputs
    hiddenDimension = 0 # Dimension of hidden layer, also the dimesnion of the encoding
    windowSize = 0 # length of window about center words to gather context
    wordCount = 0 # Number of unique words

    def __init__(self, hiddenDim, windowSize): # Initialize neural network
        self.hiddenDimension = hiddenDim
        self.windowSize = windowSize
        self.preprocess()
        self.net = ArrayNetwork(self.wordCount, 2, [self.hiddenDimension, self.wordCount], [[linear, linDerivative], [linear, linDerivative]], random=True, bias=False)

    def predict(self, word, predictionCount): # Run a center word and give most likely context
        rawOutput = self.net.evaluateNetwork(np.atleast_2d(self.centerWords[self.word2Index[word]]))
        print(word, self.centerWords[self.word2Index[word]])
        output = softmax(rawOutput)
        wordNProb = []
        for i in range(self.wordCount):
            wordNProb.append([output[0][i], self.words[i]])
        wordNProb = np.sort(wordNProb, axis=0)[::-1]
        print(wordNProb)
        outputWords = []
        for i in range(predictionCount):
            outputWords.append(wordNProb[i][1])
        return outputWords

    def writeEmbeddings(self): # Output embedding vectors into file so I can use them elsewhere
        self.net.outputEmbedding()
        with open("wordEmbedding.txt", "a") as txt_file:
            txt_file.write('[')
            for line in self.word2Index:
                txt_file.write('\'' + line + "\',\n") # works with any number of elements in a line
            txt_file.write(']')
            txt_file.close()

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

        self.words = {}
        for sentence in self.corpus:
            for token in sentence:
                if token not in word:
                    self.words[token] = 1
                else:
                    self.words[token] += 1
        self.words = sorted(list(self.words.keys()))
        #print(words)
        # Counts word frequency and wordCount in dictionary, then sorts by alphebetical order
        # I'm not sure why frequency is included, I may come back and remove that section

        self.wordCount = len(self.words)

        for i in range(len(self.words)):
            self.word2Index[self.words[i]] = i
        #print(self.word2Index)
        # Now given a word I can get its index for 1 hot encoding.

        # Add section to encode center and word stuff.
        #print(self.corpus)
        for sentence in self.corpus:
            for i in range(len(sentence)):
                center = np.zeros((self.wordCount,))
                center[self.word2Index[sentence[i]]] = 1
                context = np.zeros((self.wordCount,))
                for j in range(i - self.windowSize, i + self.windowSize + 1):
                    if (j >= 0) and (j < len(sentence)) and (j != i):
                        #print(sentence[j])
                        context[self.word2Index[sentence[j]]] += 1
                #print("Dividing line ===============")
                self.centerWords.append(center)
                self.contextWords.append(context)    
        #print(self.centerWords)
        #print('==================')
        #print(self.contextWords)
        



        

    def train(self, epochs):
        learnRate = .001
        for i in range(epochs):
            loss = 0
            for j in range(len(self.centerWords)):
                rawOutput = self.net.evaluateNetwork(np.atleast_2d(self.centerWords[j]))
                output = softmax(rawOutput)
                diffs = np.subtract(self.contextWords[j], output)
                self.net.updateNetworkGeneralizedDelta(np.atleast_2d(self.centerWords[j]), diffs, learnRate)
                for k in range(self.wordCount):
                    if self.contextWords[j][k]:
                        loss -= rawOutput[0][k]
                loss += self.wordCount * np.log(np.sum(np.exp(rawOutput)))
            loss /= len(self.centerWords)
            print("Ephoch: ", i, " Loss: ", loss)
            learnRate *= 1/((1+learnRate*i))

if __name__ == "__main__":
    model = Word2Vec2(2, 2)
    model.preprocess()
    model.train(1000)
    model.writeEmbeddings()
    print(model.predict("trump", 2))
    print(model.predict("joshua", 2))
