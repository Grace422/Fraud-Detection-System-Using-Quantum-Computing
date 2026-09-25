import pandas as pd

DATA_PATH = "data/creditcard.csv"


def main():
    print("Loading fraud detection dataset...")

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("Shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentages:")
    print(df["Class"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()