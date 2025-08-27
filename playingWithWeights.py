wordEncodings = [[-0.46124995447040645, -0.32414429116782895],
[0.34033928033625416, 0.8657305518763864],
[-0.2510124906031316, 0.2697268977626135],
[1.824602072627239, -0.5433064925650535],
[2.150861829859104, -0.9545709514512222],
[-0.12909890318677522, 0.5374182524413071],
[0.5001499184408771, -0.040489982386608656],
[0.5481402339359154, -0.4795627454996993],
[-0.8632454778620072, -0.0407901415566964],
[-1.292491844423883, 0.04301307425566612],
[-0.4206416797311197, 0.045391099841219695],
[1.01044958525552, -0.21911005892378554],
[-0.2668756965738928, -0.2538470646931593],
[0.677465107891871, 0.36689293935881095],
[-0.5418831123110094, 0.20441436746963448],
[-1.6534022398638715, 0.21715182008991266]]
wordIndices = {'abundance': 0, 'account': 1, 'accounted': 2, 'accusations': 3, 'afflicts': 4, 'against': 5, 'all': 6, 'altitude': 7, 'authority': 8, 'away': 9, 'barren': 10, 'become': 11, 'before': 12, 'bread': 13, 'caius': 14, 'cannot': 15, 'capitol': 16, 'chief': 17, 'citizen': 18, 'citizens': 19, 'city': 20, 'commonalty': 21, 'consider': 22, 'content': 23, 'corn': 24, 'could': 25, 'country': 26, 'covetous': 27, 'dear': 28, 'die': 29, 'dog': 30, 'done': 31, 'end': 32, 'enemy': 33, 'ere': 34, 'especially': 35, 'even': 36, 'famish': 37, 'famously': 38, 'faults': 39, 'first': 40, 'fort': 41, 'further': 42, 'gain': 43, 'give': 44, 'gods': 45, 'good': 46, 'guess': 47, 'hath': 48, 'hear': 49, 'help': 50, 'here': 51, 'him': 52, 'humanely': 53, 'hunger': 54, 'i': 55, 'if': 56, 'inventory': 57, 'is': 58, "is't": 59, 'kill': 60, 'know': 61, "know't": 62, 'leanness': 63, 'let': 64, 'maliciously': 65, 'marcius': 66, 'men': 67, 'might': 68, 'misery': 69, 'mother': 70, 'must': 71, 'nature': 72, 'nay': 73, 'need': 74, 'no': 75, 'not': 76, 'o': 77, 'object': 78, "on't": 79, 'one': 80, 'particularise': 81, 'partly': 82, 'patricians': 83, 'pays': 84, 'people': 85, 'pikes': 86, 'please': 87, 'poor': 88, 'prating': 89, 'price': 90, 'proceed': 91, 'proud': 92, 'rakes': 93, 'rather': 94, 'relieve': 95, 'relieved': 96, 'repetition': 97, 'report': 98, 'resolved': 99, 'revenge': 100, 'risen': 101, 'say': 102, 'second': 103, 'services': 104, 'shouts': 105, 'side': 106, 'soft-conscienced': 107, 'speak': 108, 'stay': 109, 'sufferance': 110, 'superfluity': 111, 'surfeits': 112, 'surplus': 113, 'talking': 114, 'the': 115, 'these': 116, 'think': 117, 'thirst': 118, 'though': 119, 'till': 120, 'tire': 121, 'unto': 122, 'us': 123, 'verdict': 124, 'very': 125, 'vice': 126, 'virtue': 127, 'way': 128, 'we': 129, 'well': 130, 'what': 131, 'wholesome': 132, 'word': 133, 'would': 134, 'yield': 135, 'you': 136}
import matplotlib.pyplot as plt
import numpy as np
wordX = np.transpose(wordEncodings)[0]
wordY = np.transpose(wordEncodings)[1]
labels = ['brown',
'cow',
'grey',
'hot',
'joshua',
'jumped',
'many',
'moon',
'people',
'short',
'succeeded',
'tall',
'the',
'tried',
'trump',
'ugly',
]#np.array(list(wordIndices.keys()))
#['6', 'ansley', 'bear', 'boner', 'caused', 'circus', 'delicious', 'inches', 'instead', 'joshua', 'kaya', 'like', 'long', 'ravioli', 'treated', 'unrelated']
plt.scatter(wordX, wordY)
for i, txt in enumerate(labels):
    plt.annotate(txt, (wordX[i], wordY[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()
