import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data.merge_datasets import create_empty_dataset, merge_all_datasets
from data.utils import standardise

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def load_data():
    train_df = pd.read_csv('data/split/train.csv')
    val_df = pd.read_csv('data/split/val.csv')
    test_df = pd.read_csv('data/split/test.csv')

    return train_df, val_df, test_df

def encode_labels(df, cat_cols):
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df
    
class LSTMRegressor(nn.Module):

    def __init__(self, hidden_dim, input_dim, reduce=False, embedding_dim=100):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.reduce = reduce

        if self.reduce:
            self.mlp = nn.Linear(self.input_dim, embedding_dim)
            self.input_dim = embedding_dim
            
        # The LSTM takes previous day's stock events as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.input_dim, hidden_dim)

        # The linear layer that outputs regression value for stock price change
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, yesterday):
        # self.lstm.flatten_parameters()
        if self.reduce:
            yesterday = self.mlp(yesterday)
        lstm_out, _ = self.lstm(yesterday.view(len(yesterday), 1, -1))
        out = self.linear(lstm_out.view(len(yesterday), -1))
        return out
    

if __name__ == "__main__":
    # load data
    train_df, val_df, test_df = load_data()
    # standardise columns
    train_df, val_df, _ = standardise(train_df, val_df, test_df)
    
    # define parameters for learning
    epochs = 100
    learning_rate = 0.005

    model = LSTMRegressor(hidden_dim=32, input_dim=len(train_df.columns) - 2, reduce=False).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    X, y = train_df.iloc[0:-1, :].drop(columns=["date"]), train_df["change_prev_close_percentage"].iloc[1:]
    X, y = torch.tensor(X.values, dtype=torch.float).to(device), torch.tensor(y.values, dtype=torch.float).to(device)

    # train model
    for epoch in range(epochs):
        with tqdm(total=len(X), desc ='Training - Epoch: '+str(epoch)+"/"+str(epochs), unit='chunks') as prog_bar:
            for i in range(len(X)):
                model.zero_grad()

                yesterday = X[i]
                target = y[i]

                pred = model(yesterday)
                loss = loss_function(pred, target)

                loss.backward()
                optimizer.step()
                prog_bar.set_postfix(**{'run:': "LSTM",'lr': learning_rate,
                                        'loss': f"{loss.item():.3f}"
                                        })
                prog_bar.update(1)        
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
    # evaluate model
    X_val, y_val = val_df.iloc[:, 0:-1].drop(columns=["date"]), val_df["change_prev_close_percentage"].iloc[1:]
    preds = model(torch.tensor(X_val.values, dtype=torch.float))
    loss = loss_function(preds, torch.tensor(y_val.values, dtype=torch.float))

    print(f'Validation Loss: {loss.item()}')