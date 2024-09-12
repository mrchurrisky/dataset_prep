import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

def load_parquet(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        df = pd.read_parquet(file_path)
        print("Parquet file loaded successfully.")
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"An error occurred while reading the Parquet file: {str(e)}")
        raise

def preprocess_data(df, text_column='text'):
    if text_column not in df.columns:
        raise ValueError(f"The column '{text_column}' is not in the DataFrame.")
    
    # Basic preprocessing: remove null values and convert to list
    texts = df[text_column].dropna().tolist()
    return texts

def save_dataset(texts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + '\n')

def main():
    # Load Parquet file
    file_path = './train-00000-of-00001-65b5548e09cbc609.parquet'
    df = load_parquet(file_path)

    # Preprocess data
    texts = preprocess_data(df)

    # Split into train and validation sets
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)

    # Save datasets
    save_dataset(train_texts, 'train_dataset.jsonl')
    save_dataset(val_texts, 'val_dataset.jsonl')

    print(f"Training set saved with {len(train_texts)} samples.")
    print(f"Validation set saved with {len(val_texts)} samples.")

if __name__ == "__main__":
    main()