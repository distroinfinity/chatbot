import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

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
# print(xy)
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
hidden_size = 8
output_size = len(tags)
input_size = len(XTrain[0])
learning_rate = 0.001
num_epochs = 1000

"""
print(input_size,len(allWords))
print(output_size,tags)
"""

dataset = CharDataset()
trainLoader = DataLoader(
    dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in trainLoader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        #labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')
