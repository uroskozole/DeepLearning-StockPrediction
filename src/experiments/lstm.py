import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import os
import json

from data.merge_datasets import create_empty_dataset, merge_all_datasets
from data.utils import standardise, prepare_lstm_sequence
from models.lstm import LSTMRegressor

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
    

def parse_args_():
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=5, help='Sequence length')
    parser.add_argument('--target_col', type=str, default='last_price_pozavarovalnica_sava', help='Target column')

    args = parser.parse_args()

    # epochs = args.epochs
    # learning_rate = args.learning_rate
    # seq_len = args.seq_len
    # target_col = args.target_col
    
    args.results_path = f"results/lstm/seq_len{args.seq_len}_epochs{args.epochs}_lr{args.learning_rate}/"
    os.makedirs(args.results_path, exist_ok=True)


    return args

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

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.title("Loss curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()
    plt.savefig(args.results_path + "training_loss.png")
    plt.clf()


def plot_predictions(model, X, y, val_df, args, date_index_train):
    # plot predictions for train set
    train_preds = [model(x)[-1].detach().numpy() for x in X]
    train_y = y[:, -1]
    # do reverse scaling for sensible stock prices

    date_index_val = val_df["date"]
    val_df = val_df.drop(columns=["date"])
    X_val, y_val = prepare_lstm_sequence(val_df, seq_len=args.seq_len, target_col=args.target_col)
    val_preds = [model(x)[-1].detach().numpy() for x in X_val]
    val_y = y_val[:, -1]
    
    predictions = np.append(train_preds, val_preds)
    true_values = np.append(train_y, val_y)

    plot_df = pd.DataFrame({"date": pd.concat([date_index_train.iloc[args.seq_len:], date_index_val.iloc[args.seq_len:]]), "true_values": true_values, "predictions": predictions})
    plot_df.set_index("date", inplace=True)
    plt.plot(plot_df.index, plot_df["predictions"], label="Predictions", linewidth=0.5)
    plt.plot(plot_df.index, plot_df["true_values"], label="True values", linewidth=0.5)
    plt.legend()
    plt.title("Predictions vs True values")
    plt.axvline(x=date_index_train.iloc[-1], color='b', linestyle='--')
    # make x ticks only show every 50th date
    plt.xticks(plot_df.index[::50], rotation=45)
    plt.show()

    plt.savefig(args.results_path + "predictions.png")
    
    return val_y, val_preds



def train_lstm(X, y, val_df, model, loss_function, optimizer, args):
    val_df = val_df.drop(columns=["date"])
    X_val, y_val = prepare_lstm_sequence(val_df, seq_len=args.seq_len, target_col=args.target_col)
    
    train_losses = []
    validation_losses = []
    # train model
    for epoch in range(args.epochs):
        losses = []
        with tqdm(total=len(X), desc ='Training - Epoch: '+str(epoch)+"/"+str(args.epochs), unit='chunks') as prog_bar:
            for i in range(len(X)):
                model.zero_grad()

                input = X[i]
                target = y[i]

                pred = model(input)
                loss = loss_function(pred, target)

                loss.backward()
                optimizer.step()
                prog_bar.set_postfix(**{'run:': "LSTM",'lr': args.learning_rate,
                                        'loss': f"{loss.item():.3f}"
                                        })
                prog_bar.update(1)
                losses.append(loss.item())
        
        val_preds = [model(x)[-1].detach().numpy() for x in X_val]
        val_y = y_val[:, -1]
        validation_loss = mean_squared_error(val_preds, val_y)
                
        print(f'Epoch: {epoch}, train MSE: {np.mean(np.array(losses))}, validation MSE: {validation_loss}')
        train_losses.append(np.mean(np.array(losses)))
        validation_losses.append(validation_loss)
    
    return model, train_losses, validation_losses

if __name__ == "__main__":
    # parse arguments
    args = parse_args_()
    # load data
    train_df, val_df, test_df = load_data()
    # standardise columns
    train_df, val_df, _ = standardise(train_df, val_df, test_df, args)
    
    date_index_train = train_df["date"]
    train_df = train_df.drop(columns=["date"])
    X, y = prepare_lstm_sequence(train_df, seq_len=args.seq_len, target_col=args.target_col)

    # specify the model, optimizer and loss function
    model = LSTMRegressor(hidden_dim=128, input_dim=len(train_df.columns) - 1, reduce=False)#.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    # train the model
    model, train_losses, validation_losses = train_lstm(X, y, val_df, model, loss_function, optimizer, args)
    plot_losses(train_losses, validation_losses)
    
    val_y, val_preds = plot_predictions(model, X, y, val_df, args, date_index_train)

    validation_loss = mean_squared_error(val_preds, val_y)
    print(f"Validation MSE: {validation_loss}")
    print(f"Training MSE: {train_losses[-1]}")

    results = dict()
    results["validation_loss"] = validation_loss
    results["training_loss"] = train_losses[-1]
    results["layers"] = 1
    json.dump(results, open(args.results_path + "results.json", "w"))
    

