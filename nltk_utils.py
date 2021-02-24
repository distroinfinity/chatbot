import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenizedSentence,allWords):
    """
    sentence = ["hello","how","are","you"]
    words = ["hi","hello","I","you","bye","thank","cool"]
    bag =   [ 0,   1,      0,  1,    0,    0,      0]
    """
    tokenlisedStemmedSentence = [stem(w) for w in tokenizedSentence]

    bag = np.zeros(len(allWords),dtype=np.float32)
    for idx, w in enumerate(allWords):
        if w in tokenlisedStemmedSentence:
            bag[idx] = 1.0
    
    return bag

"""
sentence = ["hello","how","are","you"]
words = ["hi","hello","I","you","bye","thank","cool"]
bag = bag_of_words(sentence,words)
print(bag)

a = "How long does it takes?"
print(a)
a = tokenize(a)
print(a)

words = ["organise","Organises","organising"]
stemmedWords = [stem(w) for w in words]
print(stemmedWords)
"""