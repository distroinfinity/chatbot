import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
allWords = data["allWords"]
tags = data["tags"]
modelState = data["modelState"]

model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(modelState)
model.eval()

botName = "Jesse"
print("Hey yo! Lets chat. Type quit to exit")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, allWords)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X).to(device)
    _, predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    #print(prob.item())
    if prob.item() >= 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: I didn;t get that ....")