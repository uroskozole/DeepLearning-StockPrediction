import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


def standardise(train_df, val_df, test_df, args):
    # standardise columns
    df_entire = pd.concat([train_df, val_df, test_df], axis=0)
    for col in df_entire.columns[1:]:
        if col == args.target_col:
            continue
        scaler = StandardScaler()
        df_entire[col] = scaler.fit_transform(df_entire[col].values.reshape(-1, 1))

    train_df = df_entire[:len(train_df)]
    val_df = df_entire[len(train_df):len(train_df) + len(val_df)]
    test_df = df_entire[len(train_df) + len(val_df):]

    return train_df, val_df, test_df

def prepare_lstm_sequence(df, seq_len, target_col, time_jump):
    X, y = [], []
    for i in range(len(df) - seq_len - time_jump + 1):
        X.append(df.iloc[i:i + seq_len].values)
        y.append(df.iloc[i+time_jump : i+seq_len+time_jump][target_col].values)
    return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)

def prepare_sequence_movement(df, seq_len, target_col, time_jump):
    X, y = [], []
    for i in range(len(df) - seq_len - time_jump + 1):
        X.append(df.iloc[i:i + seq_len].values)
        y_mov = df.iloc[i+time_jump-1 : i+seq_len+time_jump][target_col].values
        y_mov = [int(y_mov[i+1] > y_mov[i]) for i in range(len(y_mov)-1)]
        y.append(y_mov)
    return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)