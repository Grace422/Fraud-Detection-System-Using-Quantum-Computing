import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"



def load_data():
    df = pd.read_csv(DATA_PATH)

    return df


def prepare_data():

    df = load_data()

    # Select a small number of features
    features = ["V1", "V2", "V3", "V4", "Amount"]

    X = df[features]
    y = df["Class"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    print("Training samples:", X_train.shape)
    print("Testing samples:", X_test.shape)

    print("\nTraining fraud distribution:")
    print(y_train.value_counts())

    print("\nTesting fraud distribution:")
    print(y_test.value_counts())