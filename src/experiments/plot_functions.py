from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from data.utils import prepare_lstm_sequence, prepare_sequence_movement


def plot_losses(train_losses, val_losses, target_col, args, movement=False):
    # plt.figure()
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.title(f"Loss curves: {target_col}")
    plt.xlabel("Epoch")
    if movement:
        plt.ylabel("LogLoss/Accuracy")
    else:
        plt.ylabel("MSE")
    plt.legend()
    plt.savefig(args.results_path + "training_loss.png")
    if args.show_plot:
        plt.show()
    plt.clf()

def plot_predictions(model, X, y, val_df, args, date_index_train):
    # plot predictions for train set
    train_preds = [model(x)[-1].detach().numpy() for x in X]
    train_y = y[:, -1]
    # do reverse scaling for sensible stock prices

    date_index_val = val_df["date"]
    val_df = val_df.drop(columns=["date"])
    X_val, y_val = prepare_lstm_sequence(val_df, seq_len=args.seq_len, target_col=args.target_col, time_jump=args.time_jump)
    val_preds = [model(x)[-1].detach().numpy() for x in X_val]
    val_y = y_val[:, -1]
    
    predictions = np.append(train_preds, val_preds)
    true_values = np.append(train_y, val_y)

    plot_df = pd.DataFrame({"date": pd.concat([date_index_train.iloc[args.seq_len + args.time_jump - 1:], date_index_val.iloc[args.seq_len + args.time_jump - 1:]]), "true_values": true_values, "predictions": predictions})
    plot_df.set_index("date", inplace=True)
    # plt.figure()
    plt.plot(plot_df.index, plot_df["predictions"], label="Predictions", linewidth=0.5)
    plt.plot(plot_df.index, plot_df["true_values"], label="True values", linewidth=0.5)
    plt.legend()
    plt.title(f"Predictions vs True values: {args.target_col}")
    plt.axvline(x=date_index_train.iloc[-1], color='b', linestyle='--')
    # make x ticks only show every 50th date
    plt.xticks(plot_df.index[::100], rotation=45)
    plt.savefig(args.results_path + "predictions.png")
    if args.show_plot:
        plt.show()

    
    return val_y, val_preds

def plot_predictions_movement(model, X, y, val_df, args, date_index_train):
    # plot predictions for train set
    train_preds = [model(x)[-1].detach().numpy() for x in X]
    train_y = y[:, -1]
    # do reverse scaling for sensible stock prices

    date_index_val = val_df["date"]
    val_df = val_df.drop(columns=["date"])
    X_val, y_val = prepare_sequence_movement(val_df, seq_len=args.seq_len, target_col=args.target_col, time_jump=args.time_jump)
    val_preds = [model(x)[-1].detach().numpy() for x in X_val]
    val_y = y_val[:, -1]
    
    predictions = np.append(train_preds, val_preds)
    true_values = np.append(train_y, val_y)

    plot_df = pd.DataFrame({"date": pd.concat([date_index_train.iloc[args.seq_len + args.time_jump - 1:], date_index_val.iloc[args.seq_len + args.time_jump - 1:]]), "true_values": true_values, "predictions": predictions})
    plot_df.set_index("date", inplace=True)
    # plt.figure()
    plt.plot(plot_df.index, plot_df["predictions"], label="Predictions", linewidth=0.5)
    plt.plot(plot_df.index, plot_df["true_values"], label="True values", linewidth=0.5)
    plt.legend()
    plt.title(f"Predictions vs True values: {args.target_col}")
    plt.axvline(x=date_index_train.iloc[-1], color='b', linestyle='--')
    # make x ticks only show every 50th date
    plt.xticks(plot_df.index[::100], rotation=45)
    plt.savefig(args.results_path + "predictions.png")
    if args.show_plot:
        plt.show()

    
    return val_y, val_preds