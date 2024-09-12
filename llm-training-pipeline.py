import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Config, AdamW
from tqdm import tqdm

# Assuming you've already run the preprocessing script and have train_encodings and val_encodings

def create_dataloaders(train_encodings, val_encodings, batch_size=8):
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'])
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=5e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids, attention_mask = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

def main():
    # Load your preprocessed data (train_encodings and val_encodings)
    # For this example, we'll assume they're already loaded
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_encodings, val_encodings)
    
    # Initialize the model
    config = GPT2Config(vocab_size=50257, n_positions=512, n_ctx=512, n_embd=768, n_layer=6, n_head=12)
    model = GPT2LMHeadModel(config)
    
    # Train the model
    train_model(model, train_loader, val_loader)
    
    # Save the model
    model.save_pretrained("path/to/save/model")

if __name__ == "__main__":
    main()
