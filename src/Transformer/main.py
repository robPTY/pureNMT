import torch
import pandas as pd 
from typing import Tuple 
from src import DATA_DIRECTORY
from tokenizer import Tokenizer

Sets = Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
VOCAB_SIZE = 8000

def load_dataset(path: str) -> Sets:
    df = pd.read_csv(path, sep="\t", header=None, names=["english", "spanish", "meta"])
    df = df[["english", "spanish"]]

    Xs = df["english"]
    Ys = df["spanish"]

    # 80, 20 split
    vocab_size = Xs.shape[0]
    train_size = int(vocab_size * 0.80)

    X_train = Xs[:train_size]
    Y_train = Ys[:train_size]
    X_test = Xs[train_size:]
    Y_test = Ys[train_size:]

    return X_train, Y_train, X_test, Y_test

def train_tokenizer(xTrain: pd.Series, yTrain: pd.Series) -> None:
    combined_corpus = pd.concat(xTrain, yTrain)
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokens = tokenizer.tokenize(combined_corpus)
    tokenizer.save("models/tokenizer.json")

def main():
    file_path = DATA_DIRECTORY / "eng_to_spa.txt" 
    xTrain, yTrain, xTest, yTest = load_dataset(file_path)
    
    # Train tokenizer on both eng/spanish (commented out after training)
    # train_tokenizer(xTrain, yTrain)
    return 1

if __name__ == "__main__":
    main()