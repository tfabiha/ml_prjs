"""
Emotion Classification with Neural Networks

Extensions:
learning rate scheduler (located here) and CNN model (located in models.py)
"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
import argparse

# Global definitions - data
DATA_FN = 'data/crowdflower_emotion.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator, has_scheduler = False, scheduler = None):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :param has_scheduler: a boolean that is true when we are using a larning scheduler for optimization
    :param scheduler: containers a scheduler object if we are using a learning scheduler or else is None
    :return model, the trained model
    """

    for param in model.parameters():
        param.requires_grad = True

    # general loss for the first epoch so that we always decreasing the loss the first time we learn
    prev_loss = 10000000
    
    times_increase_loss = 0

    epoch = 0

    # we can only increase (or stay at the same) loss five times before we decide that we have stopped learning
    # this is not consequtive increase
    # i believe this is the best method because the loss will sometimes increase as the optimizer attempts to find
    # the optimum values. by setting a limit on how many times we can do that we assure that once it stops decreasing
    # loss we will have already exited the training phase
    while times_increase_loss < 5:
        # train model in batches
        for train_data, train_labels in train_generator:
            model.zero_grad()
            train_pred = model(train_data)
            loss = loss_fn(train_pred.double(), train_labels.long())

            loss.backward()
            optimizer.step()
        
        # at the end of an epoch update the scheduler if we are using extension1
        if has_scheduler:
            scheduler.step()

        gold = []
        predicted = []

        loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
        if USE_CUDA:
            loss = loss.cuda()

        with torch.no_grad():
            for dev_data, dev_labels in dev_generator:
                model.zero_grad()
                dev_pred = model(dev_data)

                gold.extend(dev_labels.cpu().detach().numpy())
                predicted.extend(dev_pred.argmax(1).cpu().detach().numpy())

                loss += loss_fn(dev_pred.double(), dev_labels.long())

        if prev_loss <= loss:
            times_increase_loss += 1
        prev_loss = loss
        epoch += 1

        print("times_increase_loss",times_increase_loss)
        print("Epoch",epoch)
        print("Dev loss: ")
        print(loss)
        print("F-score: ")
        print(f1_score(gold, predicted, average='macro'))
        print("")


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
            

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    if args.model == 'extension2':
        model = models.ExperimentalNetwork(embeddings)
    elif args.model == 'RNN':
        model = models.RecurrentNetwork(embeddings)
    else:
        # do the dense model if dense or extension1
        model = models.DenseNetwork(embeddings)
        
    if args.model == 'extension1':
        # made scheduler for extension-grading
        # this is extension 1
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        lambda1 = lambda epoch: 0.65 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        train_model(model, loss_fn, optimizer, train_generator, dev_generator, True, scheduler)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, loss_fn, optimizer, train_generator, dev_generator, False, None)

    test_model(model, loss_fn, test_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)

    """
    I used a learning rate scheduler for my first extension. I thought it would be best to add this in as an extension so 
    that the models would be able to more quickly find the optimal weights. Since the beginning has us starting out with 
    random weights it made the most sense to have a higher learning rate like 0.01 and then using a scheduling rate that 
    says we take 65% of the previous learning rate for our current learning rate. I initialized the scheduler in the main 
    function, and called torch.optim.lr scheduler to obtain the scheduler. In order to pass this on to actually change the 
    learning rate though I had to pass it onto the training function by changing the signature to train model(model, loss 
    fn, optimizer, train generator, dev generator, has scheduler = False, scheduler = None). Adding has scheduler, scheduler 
    allowed me to call schedule.step() after each epoch. Adding in the scheduler however didn’t really improve performance. 
    My dense model already had a testing f1 macro score of .446 and the scheduler changed this to .442, a negligible difference. 
    It might have been that the learning rate was too small to get out of the localized optimum that was found.

    I used a CNN model for my second extension because I wanted to test how a convolution algorithm would compare to a recurrent 
    algorithm. I assumed that by doing convolutions we might be able to understand some hidden under- lying patterns in the word 
    embeddings per sentence. My changes were made directly in model.py under the ExperimentalNetwork class. This model ended up 
    slightly worse than the RNN model, since after the training phase the RNN model had an f1 macro score of 0.430 while the CNN 
    model had an f1 macro score of 0.405. This might be because CNN performs better at looking at the whole picture while RNN 
    uses sequential information, and thus can predict emotions better since the sentences come in a sequence.
    """
