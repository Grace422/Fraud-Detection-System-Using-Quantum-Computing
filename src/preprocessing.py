import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"


FEATURES = ["V1", "V2", "V3", "V4", "Amount"]


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data():

    df = load_data()

    X = df[FEATURES]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def prepare_quantum_data():

    df = load_data()

    # Separate legitimate and fraudulent transactions
    legitimate = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    # Use the same number of legitimate transactions as fraud transactions
    legitimate_sample = legitimate.sample(
        n=len(fraud),
        random_state=42
    )

    # Combine the two classes
    quantum_df = pd.concat(
        [legitimate_sample, fraud]
    )

    # Shuffle
    quantum_df = quantum_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    X = quantum_df[FEATURES]
    y = quantum_df["Class"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    print("CLASSICAL DATA")
    print("Training:", X_train.shape)
    print("Testing :", X_test.shape)

    X_train_q, X_test_q, y_train_q, y_test_q = prepare_quantum_data()

    print("\nQUANTUM DATA")
    print("Training:", X_train_q.shape)
    print("Testing :", X_test_q.shape)

    print("\nQuantum training distribution:")
    print(y_train_q.value_counts())

    print("\nQuantum testing distribution:")
    print(y_test_q.value_counts())