"""
Emotion Classification with Neural Networks - Models File
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class DenseNetwork(nn.Module):
    def __init__(self, embeddings):
        super(DenseNetwork, self).__init__()
        
        # the hidden layer parameter
        self.hnum = 70
        
        # create the embedding layer setup
        vocab_length, embedding_dim = embeddings.size()
        emb_layer = nn.Embedding(vocab_length, embedding_dim)
        emb_layer.weight = nn.Parameter(embeddings)
        
        self.embedding = emb_layer
        
        # the two layer feed forward network setup
        self.fc1 = nn.Linear(embedding_dim, self.hnum)
        self.fc2 = nn.Linear(self.hnum, 4)
        
    def forward(self, x):
        
        # embed the input
        embeds = self.embedding(x)
        
        # collect the sum of all word embeddings in a sentence
        # then flatten
        sum_embeds = torch.sum(embeds, dim=1)
        sum_embeds = sum_embeds.view(sum_embeds.size(0), -1).float()
        
        # run the embeds through the first hidden layer
        # perform relu activation function
        first_nodes = self.fc1(sum_embeds)
        activated_node = nn.functional.relu(first_nodes)
        
        # run through the second hidden layer
        second_nodes = self.fc2(activated_node)
        
        # soft max all the inputs in the end to normalize
        prediction = nn.functional.softmax(second_nodes, dim=1)
        
        return prediction
    
class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings):
        super(RecurrentNetwork, self).__init__()
        
        # the hidden layer parameter
        self.hnum = 90
        
        # create the embedding layer setup
        vocab_length, embedding_dim = embeddings.size()
        emb_layer = nn.Embedding(vocab_length, embedding_dim)
        emb_layer.weight = nn.Parameter(embeddings)
        
        self.embedding = emb_layer
        
        # the rnn - uses 2 layers
        self.rnn = nn.RNN(embedding_dim, self.hnum, num_layers=2, batch_first = True)
        
        # linear feed forward network to get to our output size
        self.fc1 = nn.Linear(self.hnum, 4)
    
    # takes each sentence in s and finds the length of it without the 0 padding
    def computelengths(self, s):
        lengths = []
        for sent in s:
            try:
                # find the index at which we see a 0 in the sentence
                # that is the length of the sentence
                lengths.append((sent == 0).nonzero(as_tuple=True)[0][0])
            except:
                # this occurs when we fail to find a 0 in the list
                # this means that the length of the sentence is equal to the size of the sentence
                lengths.append(sent.size()[0])
    
        return lengths
    
    # x is a PaddedSequence for an RNN
    def forward(self, x):
        
        # get the lengths of all the sentences in x
        lengths = self.computelengths(x)
        
        # embed the input x
        embeds = self.embedding(x)
        embeds = embeds.float()
        
        # pack the embeds so that the rnn doesn't calculate based on the 0 paddings
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        
        unpacked, hidden = self.rnn(packed)
        
        # since the hidden layer stores the information gained at the end of each sentence in the previous iterations
        # we can use the last hidden state to compute our answer
        hidden = hidden[-1, :, :]
        
        # run through a linear layer to make prediction for each sentence
        first_nodes = self.fc1(hidden)
        
        # soft max in the end to normalize
        prediction = nn.functional.softmax(first_nodes, dim=1)
        
        return prediction

# made cnn for extension-grading
# this is extension 2
class ExperimentalNetwork(nn.Module):
    def __init__(self, embeddings):
        super(ExperimentalNetwork, self).__init__()
        
        # the hidden layer parameter
        self.hnum1 = 32
        self.hnum2 = 64
        self.hnum3 = 70
        
        # create the embedding layer setup
        vocab_length, embedding_dim = embeddings.size()
        emb_layer = nn.Embedding(vocab_length, embedding_dim)
        emb_layer.weight = nn.Parameter(embeddings)
        
        self.embedding = emb_layer
        
        # 2 layer cnn
        self.c1 = nn.Sequential(
            nn.Conv1d(embedding_dim, self.hnum1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.c2 = nn.Sequential(
            nn.Conv1d(self.hnum1, self.hnum2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        # 2 layer feed forward system
        self.fc1 = nn.Linear(self.hnum2, self.hnum3)
        self.fc2 = nn.Linear(self.hnum3, 4)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        
        # embed the input x
        embeds = self.embedding(x)
        embeds = embeds.float()
        
        # change the shape of embeds so it can be passsed into the first cnn layer
        embeds = embeds.permute(0, 2, 1)
        
        # pass thorugh both cnn layers
        out = self.c1(embeds)
        out = self.c2(out)
        
        # flatten it out so we can pass this through the forward layer
        out = torch.sum(out, dim=2)
        
        # pass thorugh the feed forward layers
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        
        return out
