import pandas as pd


# Drops majority null features
def drop_null_majority_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    cols_to_drop = []
    for col in df_train.columns:
        if df_train[col].isnull().sum() > len(df_train[col])/2:
            cols_to_drop.append(col)

    df_train.drop(cols_to_drop, axis=1, inplace=True)
    df_test.drop(cols_to_drop, axis=1, inplace=True)


# Drops features with single value majority (above 80%)
def drop_single_val_majority(df_train: pd.DataFrame, df_test: pd.DataFrame):
    cols_to_drop = []
    for col in df_train.columns:
        var_count = df_train[col].value_counts() / df_train.shape[0]
        for key in var_count.keys():
            if var_count[key] > 0.80:
                cols_to_drop.append(col)

    # print(cols_to_drop)
    df_train.drop(cols_to_drop, axis=1, inplace=True)
    df_test.drop(cols_to_drop, axis=1, inplace=True)
