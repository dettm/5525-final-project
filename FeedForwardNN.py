import torch
import torch.nn as nn
from torch import optim
import random

class NeuralSentimentClassifier(nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    def __init__(self, hid, embedding_length):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param hid: size of hidden layer(integer)
        :param embedding_length: size of the input embeddings
        """
        #super(nn.Module, self).__init__()
        super().__init__()

        self.V = nn.Linear(embedding_length, hid)
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, 2)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Runs the neural network on the given data and returns log probabilities of the two classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        # print("log probs = " + repr(self.log_softmax(self.W(self.g(self.V(x))))))
        return self.log_softmax(self.W(self.g(self.V(x))))

    def predict(self, x: torch.tensor) -> torch.tensor:
        """
        Predicts the output of a single input example x

        :param x: an input example, dimensionality of 1
        :return: the predicted class for the input example
        """
        return torch.argmax(self(x)).item()


def train_feed_forward_classifier(train_exs, train_ys, n_epochs=180, h_dim=100, init_lr=0.0005, b_size=20) -> NeuralSentimentClassifier:
    """
    :param train_exs: the training set, in tensor form
    :param train_ys: the training set labels, in tensor form
    :param n_epochs: the number of epochs
    :param b_size: batch size
    :param init_lr: the initial learning rate passed to the optimizer
    :param h_dim: the dimensionality of the hidden layer
    :return: A trained NeuralSentimentClassifier model
    """

    # INITIALIZATION
    num_epochs = n_epochs
    hidden1_dims = h_dim
    initial_learning_rate = init_lr
    batch_size = b_size
    print("===BEGIN TRAINING===")
    print("Hidden dimension: " + repr(hidden1_dims))
    print("# Epochs: " + repr(num_epochs))
    print("lr: " + repr(initial_learning_rate))
    print("batch size: " + repr(batch_size))
    print("embedding length: " + repr(len(train_exs[0])))
    print("====================")
    sc = NeuralSentimentClassifier(hidden1_dims, len(train_exs[0]))
    optimizer = optim.Adam(sc.parameters(), lr=initial_learning_rate)

    # TRAINING
    for epoch in range(num_epochs):
        ex_indices = [i for i in range(len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        x_batch = torch.empty((batch_size, len(train_exs[0])))
        batch_labels = torch.empty(batch_size)
        for i in range(len(ex_indices) // batch_size):
            for j, idx in enumerate(ex_indices[i*batch_size:(i+1)*batch_size]):
                x_batch[j] = train_exs[idx]
                batch_labels[j] = train_ys[idx].item()
            batch_labels = batch_labels.long()

            # Zero out the gradients from the FFNN object.
            sc.zero_grad()
            log_probs = sc(x_batch)
            # Compute loss
            loss = nn.NLLLoss()
            l = loss(log_probs, batch_labels)
            total_loss += l
            # Computes the gradient and takes the optimizer step
            l.backward()
            optimizer.step()
        #print("Total loss on epoch %i: %f" % (epoch+1, total_loss))

    return sc


def eval_nn(ffnn, test_x, test_y):
    """
    Evaluation function intended to be used to evaluate the accuracy and f1 score (harmonic mean of precision and
    recall) of a trained neural network. For our task the f1 score is especially important

    :param ffnn: the trained feed-forward neural network
    :param test_x: the data that the neural network will be evaluated on
    :param test_y: the gold labels for the test_x dataset
    :return: accuracy and f1, where the values are the measured values for the accuracy of the model and the f1 score
    for the spam class
    """
    n_correct = 0
    n_pos_correct = 0
    num_pred = 0
    num_gold = 0
    for i in range(len(test_x)):
        gold_label = test_y[i]
        pred = ffnn.predict(test_x[i])
        if pred == gold_label:
            n_correct += 1
        if pred == 1:
            num_pred += 1
        if gold_label == 1:
            num_gold += 1
        if pred == 1 and gold_label == 1:
            n_pos_correct += 1
    acc = float(n_correct) / len(test_x)
    prec = float(n_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(n_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0

    return acc, f1
