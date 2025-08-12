from django.shortcuts import render
import torch
import random
#import render
import json
import nltk
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .chatbot_model import NeuralNet

nltk.download('punkt')
# Define stemmer
stemmer = nltk.stem.PorterStemmer()

# Define tokenization function
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Define stemming function
def stem(word):
    return stemmer.stem(word.lower())

# Define bag of words function
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

with open('chatbot_app\intents.json', 'r') as f:
    intents = json.load(f)

def chatbot_response(user_input, model, all_words, tags):
    user_input = nltk.word_tokenize(user_input)
    X = bag_of_words(user_input, all_words)
    X = X.reshape(1, X.shape[0]) # Reshape to match the model's input shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.from_numpy(X).to(device)

    # Get the model's prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

 
    response = []
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = intent['response']

    return random.choice(response)

def chatbot(request):
    if request.method == 'POST':
        user_input = request.POST['user_input']

        # Load your model and data
        data = torch.load("chatbot_app\data.pth")
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval()

        # Process user input and generate a response
        response_from_chatbot = chatbot_response(user_input, model, all_words, tags)

        return render(request, 'chatbot.html', {'user_input': user_input, 'response_from_chatbot': response_from_chatbot})

    return render(request, 'chatbot.html')
