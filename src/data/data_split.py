import pandas as pd

from data.merge_datasets import create_empty_dataset, merge_all_datasets, correct_dtypes


TRAIN_SIZE = .666
VAL_SIZE = (1 - TRAIN_SIZE) / 2
TEST_SIZE = VAL_SIZE

def make_data_split():
    # load data
    df = create_empty_dataset()
    df = merge_all_datasets(df)
    df = correct_dtypes(df, df.columns[1:])
    
    df = df.fillna(-1)

    # split data
    train_df = df[:int(len(df) * TRAIN_SIZE)]
    val_df = df[int(len(df) * TRAIN_SIZE):int(len(df) * (TRAIN_SIZE + VAL_SIZE))]
    test_df = df[int(len(df) * (TRAIN_SIZE + VAL_SIZE)):]

    return train_df, val_df, test_df

def save_data_split(train_df, val_df, test_df):

    train_df.to_csv('data/split/train.csv', index=False)
    val_df.to_csv('data/split/val.csv', index=False)
    test_df.to_csv('data/split/test.csv', index=False)    


if __name__ == '__main__':
    train_df, val_df, test_df = make_data_split()
    save_data_split(train_df, val_df, test_df)
