import pandas as pd


# Drops majority null features
def drop_null_majority_features(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col='SalePrice'):
    features_to_drop = []
    features_created = []

    for col in df_train.columns:
        if col == target_col:
            continue
        missing_pct = df_train[col].isnull().sum() / len(df_train)

        if missing_pct > 0.75:
            features_to_drop.append(col)
        elif missing_pct > 0.15:
            features_created.append(col)

    print("Analyzing single-value dominant features:")
    for col in df_train.columns:
        if col == target_col or col in features_to_drop:
            continue

        if df_train[col].dtype in ['object', 'category']:
            # For categorical
            value_counts = df_train[col].value_counts(normalize=True, dropna=False)
            if len(value_counts) > 0 and value_counts.iloc[0] > 0.95:
                features_to_drop.append(col)
                print(f"  DROPPING {col}: {value_counts.iloc[0]*100:.1f}% are '{value_counts.index[0]}'")
        else:
            # For numeric
            if df_train[col].nunique() == 1:  # Truly constant
                features_to_drop.append(col)
                print(f"  DROPPING {col}: Constant value")

    df_train.drop(features_to_drop, axis=1, inplace=True)
    df_test.drop(features_to_drop, axis=1, inplace=True)

    return features_created
