import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import re

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text.strip()

def preprocess_data(data, text_column, tokenizer, max_length=512):
    cleaned_texts = [clean_text(item[text_column]) for item in data]
    
    # Tokenize the texts
    encodings = tokenizer(cleaned_texts, truncation=True, padding='max_length', 
                          max_length=max_length, return_tensors='pt')
    
    return encodings

def main():
    # Load the data
    data = load_data('path/to/your/converted_file.json')
    
    # Split the data
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Initialize tokenizer (replace 'gpt2' with your chosen model)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Preprocess the data
    train_encodings = preprocess_data(train_data, 'text_column', tokenizer)
    val_encodings = preprocess_data(val_data, 'text_column', tokenizer)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Example encoding: {train_encodings['input_ids'][0][:10]}")

if __name__ == "__main__":
    main()
