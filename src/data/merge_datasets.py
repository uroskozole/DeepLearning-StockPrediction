import pandas as pd
import os


ETF_PATH = "data/ETF"
INDEX_PATH = "data/Indexes"
SECURITIES_PATH = "data/securities"
STOCK_PATH = "data/stock_movement"

def create_empty_dataset():
    dates = pd.date_range(start='2018-01-01', end='2024-06-05', freq='D')
    df = pd.DataFrame(dates, columns=['date'])
    # drop weekends
    df = df[df['date'].dt.dayofweek < 5]
    return df

def merge_datasets(df1, df2, suffix, securities=False):
    if securities:
        df = pd.merge(df1, df2, left_on='date', right_on='segment_listing_date', how='left', suffixes=('', "_" + suffix))
        df = df.drop(columns=['segment_listing_date'])
        df = df.sort_values(by='date')
        
        
        return df

    else:
        df = pd.merge(df1, df2, on='date', how='left', suffixes=('', "_" + suffix))
        df = df.sort_values(by='date')

        df[f"last_price_{suffix}"] = df[f"last_price_{suffix}"].ffill()
        df["open_price_" + suffix] = df["open_price_" + suffix].fillna(df[f"last_price_{suffix}"])
        df["high_price_" + suffix] = df["high_price_" + suffix].fillna(df[f"last_price_{suffix}"])
        df["low_price_" + suffix] = df["low_price_" + suffix].fillna(df[f"last_price_{suffix}"])
        if "vwap_price_" + suffix in df.columns: # indexes do not have vwap_price
            df["vwap_price_" + suffix] = df["vwap_price_" + suffix].fillna(df[f"last_price_{suffix}"])
        
        df["change_prev_close_percentage_" + suffix] = df["change_prev_close_percentage_" + suffix].fillna(0)
        if "num_trades_" + suffix in df.columns: # indexes do not have num_trades
            df["num_trades_" + suffix] = df["num_trades_" + suffix].fillna(0)
        if "volume_" + suffix in df.columns:
            df["volume_" + suffix] = df["volume_" + suffix].fillna(0)
        df["turnover_" + suffix] = df["turnover_" + suffix].fillna(0)

        return df

def merge_all_datasets(empty):
    dfs = []
    names = []

    # import datasets from ETF_PATH
    for file in os.listdir(ETF_PATH):
        df = pd.read_csv(os.path.join(ETF_PATH, file), delimiter=';')
        df = df.drop(columns=['mic', 'symbol', 'trading_model_id', 'isin', 'price_currency', 'turnover_currency'])
        dfs.append(df)
        names.append(file.split('.')[0])

    # import datasets from INDEX_PATH
    for file in os.listdir(INDEX_PATH):
        df = pd.read_csv(os.path.join(INDEX_PATH, file), delimiter=';')
        df = df.drop(columns=['isin', 'mic', 'symbol'])
        dfs.append(df)
        names.append(file.split('.')[0])

    # import datasets from SECURITIES_PATH
    for file in os.listdir(SECURITIES_PATH):
        df = pd.read_csv(os.path.join(SECURITIES_PATH, file), delimiter=';')
        df = df.drop(columns=['symbol', 'name', 'sector_id', 'model', 'segment', 'security_class', 'security_type', 'isin', 'segment_delisting_date', 'debt_maturity_date', 'nominal_currency_id'])
        dfs.append(df)
        names.append(file.split('.')[0])

    # import datasets from STOCK_PATH
    for file in os.listdir(STOCK_PATH):
        df = pd.read_csv(os.path.join(STOCK_PATH, file), delimiter=';')
        df = df.drop(columns=['mic', 'symbol', 'trading_model_id', 'isin', 'price_currency', 'turnover_currency'])
        dfs.append(df)
        names.append(file.split('.')[0])

    for i in range(len(dfs)):
        df = merge_datasets(df, dfs[i], suffix=names[i], securities=names[i] == 'securities_issued')

    return df

def correct_dtypes(df, num_cols):
    for col in num_cols:
        # df[col] = pd.to_numeric(df[col], errors='coerce')
        # if dtype is object, cast it string, replace commas with dots and cast it to float
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(',', '.')
            df[col] = df[col].astype(float)
    return df



if __name__ == '__main__':
    df = create_empty_dataset()
    df = merge_all_datasets(df)
    df = correct_dtypes(df, df.columns[1:])
    
    df = df.fillna(-1)

    print(df.shape)