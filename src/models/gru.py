import torch
import torch.nn as nn



class GRURegressor(nn.Module):

    def __init__(self, hidden_dim, input_dim, reduce=False, embedding_dim=100):
        super(GRURegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.reduce = reduce

        if self.reduce:
            self.mlp = nn.Linear(self.input_dim, embedding_dim)
            self.input_dim = embedding_dim
            
        # The LSTM takes previous day's stock events as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru = nn.GRU(self.input_dim, hidden_dim)

        # The linear layer that outputs regression value for stock price change
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, yesterday):
        # self.lstm.flatten_parameters()
        if self.reduce:
            yesterday = self.mlp(yesterday)
        gru_out, _ = self.gru(yesterday)
        #out = self.linear(lstm_out)
        out = self.linear(gru_out.view(len(yesterday), -1)).squeeze()
        return out
    

class GRUMovement(nn.Module):

    def __init__(self, hidden_dim, input_dim, reduce=False, embedding_dim=100):
        super(GRUMovement, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.reduce = reduce

        if self.reduce:
            self.mlp = nn.Linear(self.input_dim, embedding_dim)
            self.input_dim = embedding_dim
            
        # The LSTM takes previous day's stock events as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru = nn.GRU(self.input_dim, hidden_dim)

        # The linear layer that outputs regression value for stock price change
        self.linear = nn.Linear(hidden_dim, 1)

        # finally the sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, yesterday):
        # self.lstm.flatten_parameters()
        if self.reduce:
            yesterday = self.mlp(yesterday)
        gru_out, _ = self.gru(yesterday)
        #out = self.linear(lstm_out)
        out = self.linear(gru_out.view(len(yesterday), -1)).squeeze()
        out = self.sigmoid(out)
        return out
    