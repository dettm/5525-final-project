import torch
import torch.nn as nn
from torch import optim
import random


def train_rnn_classifier(train_exs, train_labels, e_dim, n_epochs=10, b_size=1, h_dim=25, init_lr=0.001, n_layers=1):
    """
    :param train_exs: tensor of strings followed by consonants
    :param train_labels: tensor of labels for the training examples
    :param e_dim: dimensionality of the embeddings
    :param n_epochs: number of epochs for training
    :param b_size: batch size
    :param h_dim: the dimensionality of the hidden layer
    :param init_lr: the initial learning rate passed to the optimizer
    :param n_layers: number of layers in the LSTM
    :return: an RNNClassifier instance trained on the given data
    """
    # INITIALIZATION
    num_epochs = n_epochs
    hidden1_dims = h_dim
    embedding_dim = e_dim
    initial_learning_rate = init_lr
    batch_size = b_size
    num_layers = n_layers
    print("Embedding dimension: " + repr(embedding_dim))
    print("Hidden dimension: " + repr(hidden1_dims))
    print("# Epochs: " + repr(num_epochs))
    print("lr: " + repr(initial_learning_rate))
    print("batch size: " + repr(batch_size))
    print("# Layers in LSTM: " + repr(num_layers))

    # c = nn.init.xavier_uniform_(torch.empty(num_layers, batch_size, hidden1_dims))
    # h = nn.init.xavier_uniform_(torch.empty(num_layers, batch_size, hidden1_dims))
    rnn = RNNClassifier(train_exs, hidden1_dims, embedding_dim, num_layers)
    optimizer = optim.Adam(rnn.parameters(), lr=initial_learning_rate)

    # training loop
    for epoch in range(num_epochs):
        ex_indices = [i for i in range(len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for i in range(len(ex_indices) // batch_size):
            x_batch = train_exs[i*batch_size:(i+1)*batch_size]
            batch_labels = train_labels[i*batch_size:(i+1)*batch_size]

            # Zero out the gradients from the FFNN object.
            optimizer.zero_grad()
            log_probs, new_state = rnn(x_batch)
            # Make predictions and compute loss
            loss = nn.NLLLoss()
            l = loss(torch.unsqueeze(log_probs, 0), batch_labels)
            total_loss += l
            # Computes the gradient and takes the optimizer step
            l.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch+1, total_loss))

    return rnn


class RNNClassifier(nn.Module):

    def __init__(self, train_exs, h_dim, e_dim, num_layers):
        super().__init__()

        self.EMBEDDING_DIM = e_dim
        self.HIDDEN_DIM = h_dim
        self.NUM_LAYERS = num_layers

        self.train_exs = train_exs

        # initialize the layers
        self.rnn = nn.LSTM(self.EMBEDDING_DIM, self.HIDDEN_DIM, self.NUM_LAYERS)

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        # go from output of rnn to softmax
        self.w = nn.Linear(self.HIDDEN_DIM, 2)
        nn.init.xavier_uniform_(self.w.weight)
        self.log_softmax = nn.LogSoftmax()

    # def predict(self, context):
    #     """
    #     :param context: a single string example to be classified
    #     :return: a tensor containing the predicted classes for each sample
    #     """
    #
    #     # create the input tensor
    #     inpt = []
    #     for let in context:
    #         inpt += [self.vocab_index.index_of(let)]
    #     inpt_tens = torch.tensor(inpt)
    #     inpt_tens = torch.unsqueeze(inpt_tens, 1)
    #
    #     res = self(inpt_tens)
    #     res1 = res[0]
    #     res3 = torch.argmax(res1).item()
    #     return res3

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Runs the neural network on the given data and returns log probabilities of the two classes.

        :param x: input data with size (sequence_len, batch_size)
        :param s: a tuple of two tensors that represent the state of the rnn
        :return: an [out]-sized tensor of log probabilities
        """
        # (num_layers, batch_size, hidden_dim)
        c = nn.init.xavier_uniform_(torch.zeros(1, 1, 25))
        h = nn.init.xavier_uniform_(torch.zeros(1, 1, 25))
        rnn_out, new_s = self.rnn(x, (h, c))
        w = self.w(new_s[0][0][0])
        ls = self.log_softmax(w)
        return ls, new_s