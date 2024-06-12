import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from data.utils import standardise, prepare_sequence_movement
from models.gru import GRUMovement
from experiments.plot_functions import plot_losses, plot_predictions_movement

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
    

def parse_args_():
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=7, help='Sequence length')
    parser.add_argument('--target_col', type=str, default='last_price_krka', help='Target column')
    parser.add_argument('--scheduler', action="store_true", help='Use learning rate scheduler')
    parser.add_argument('--time-jump', type=int, default=1, help='Time jump for predictions')
    parser.add_argument('--show-plot', action="store_true", help='Show plots of predictions and losses')

    args = parser.parse_args()
    
    args.results_path = f"results/movement/gru/{args.time_jump}/seq_len{args.seq_len}_epochs{args.epochs}_lr{args.learning_rate}_scheduler{args.scheduler}/"
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

def train_lstm(X, y, val_df, model, loss_function, optimizer, args):
    val_df = val_df.drop(columns=["date"])
    X_val, y_val = prepare_sequence_movement(val_df, seq_len=args.seq_len, target_col=args.target_col, time_jump=args.time_jump)
    
    train_losses = []
    validation_losses = []

    if args.scheduler:
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
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
        if args.scheduler:
            scheduler.step()

        val_preds = [int(model(x)[-1].detach().numpy() > 0.4999999) for x in X_val]
        val_y = y_val[:, -1]
        validation_score = np.mean(np.array(val_preds) == val_y.detach().numpy())
                
        print(f'Epoch: {epoch}, train MSE: {np.mean(np.array(losses))}, validation accuracy: {validation_score}')
        train_losses.append(np.mean(np.array(losses)))
        validation_losses.append(validation_score)
    
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
    X, y = prepare_sequence_movement(train_df, seq_len=args.seq_len, target_col=args.target_col, time_jump=args.time_jump)

    # specify the model, optimizer and loss function
    model = GRUMovement(hidden_dim=128, input_dim=len(train_df.columns), reduce=False)#.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.BCELoss()

    # train the model
    model, train_losses, validation_losses = train_lstm(X, y, val_df, model, loss_function, optimizer, args)
    plot_losses(train_losses, validation_losses, args.target_col, args, movement=True)
    
    val_y, val_preds = plot_predictions_movement(model, X, y, val_df, args, date_index_train)

    validation_loss = np.min(validation_losses)
    train_loss_val_min = train_losses[np.argmin(validation_losses)]

    print(f"Validation accuracy: {validation_loss}")
    print(f"Training logloss: {train_losses[-1]}")

    results = dict()
    results["validation_accuracy"] = float(validation_loss)
    results["training_loss"] = train_losses[-1]
    results["layers"] = 1
    results["opt_epochs"] = int(np.argmax(validation_losses))
    json.dump(results, open(args.results_path + "results.json", "w"))
    

