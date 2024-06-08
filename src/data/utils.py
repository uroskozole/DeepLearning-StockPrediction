import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardise(train_df, val_df, test_df):
    # standardise columns
    df_entire = pd.concat([train_df, val_df, test_df], axis=0)
    scaler = StandardScaler()
    for col in df_entire.columns[1:]:
        df_entire[col] = scaler.fit_transform(df_entire[col].values.reshape(-1, 1))

    train_df = df_entire[:len(train_df)]
    val_df = df_entire[len(train_df):len(train_df) + len(val_df)]
    test_df = df_entire[len(train_df) + len(val_df):]

    return train_df, val_df, test_df