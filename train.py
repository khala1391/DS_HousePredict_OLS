import numpy as np
import pandas as pd
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

DATA_PATH = "data/bengaluru_house_prices.csv"


def convert_sqft_to_num(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2
        except:
            return None
    try:
        return float(x)
    except:
        return None


def preprocess(df):
    df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
    df = df.dropna()

    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df = df[df['total_sqft'].notnull()]

    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

    df['location'] = df['location'].apply(lambda x: x.strip())
    location_stats = df['location'].value_counts()
    rare_locations = location_stats[location_stats <= 10].index
    df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)

    # Remove sqft < 300 per bhk
    df = df[~(df['total_sqft'] / df['bhk'] < 300)]

    # Remove price_per_sqft outliers per location
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced], ignore_index=True)
    df = df_out

    # Remove bath outliers
    df = df[df['bath'] < df['bhk'] + 2]

    df = df.drop(['size', 'price_per_sqft'], axis='columns')
    return df


def build_features(df):
    dummies = pd.get_dummies(df['location'])
    df = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
    df = df.drop('location', axis='columns')
    return df


def train():
    df = pd.read_csv(DATA_PATH)
    print(f"Raw data: {df.shape}")

    df = preprocess(df)
    print(f"After preprocessing: {df.shape}")

    df = build_features(df)

    X = df.drop('price', axis='columns')
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    model = LinearRegression()
    model.fit(X_train, y_train)

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"CV scores: {scores}")
    print(f"Mean CV score: {scores.mean():.4f}")
    print(f"Test score (R2): {model.score(X_test, y_test):.4f}")

    # Save model, columns, and eval data for visualization
    import os; os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    columns = {"data_columns": list(X.columns)}
    with open("model/columns.json", "w") as f:
        json.dump(columns, f)
    joblib.dump({"y_test": y_test.values, "y_pred": model.predict(X_test)}, "model/eval_data.pkl")

    print("Saved: model.pkl, columns.json, eval_data.pkl")
    return model, list(X.columns)


if __name__ == "__main__":
    train()
