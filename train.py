import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as f:
    intents = json.load(f)

allWords = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        allWords.extend(w)
        xy.append((w, tag))

ignoreWords = ['?', '!', '.', ',']
allWords = [stem(w) for w in allWords if w not in ignoreWords]
allWords = sorted(set(allWords))
tags = sorted(set(tags))
# print(allWords)
# print(tags)
print(xy)
XTrain = []
YTrain = []
for (patternSentence, tag) in xy:
    bag = bag_of_words(patternSentence, allWords)
    XTrain.append(bag)

    label = tags.index(tag)
    YTrain.append(label)

XTrain = np.array(XTrain)
YTrain = np.array(YTrain)


class CharDataset(Dataset):
    def __init__(self):
        self.NSamples = len(XTrain)
        self.XData = XTrain
        self.YData = YTrain

    # dataset[idx]
    def __getitem__(self, index):
        return self.XData[index], self.YData[index]

    def __len__(self):
        return self.NSamples


# Hyperparameters
batchSize = 8

dataset = CharDataset()
trainLoader = DataLoader(
    dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=2)
