import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class ChatDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        encoded = self.tokenizer.encode_plus(
            conversation,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten()
        }

def prepare_data(file_path):
    # Load and preprocess your data
    # This is a placeholder - you'll need to implement this based on your data format
    with open(file_path, 'r') as f:
        conversations = f.readlines()
    return conversations

def train(model, train_dataloader, val_dataloader, epochs, device, lr=2e-5):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_eval_loss = 0
        for batch in tqdm(val_dataloader, desc="Validation"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} - Average validation loss: {avg_val_loss:.4f}")

    return model

def main():
    # Load the pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare your data
    conversations = prepare_data('path/to/your/chat_data.txt')

    # Create datasets
    full_dataset = ChatDataset(conversations, tokenizer)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = train(model, train_dataloader, val_dataloader, epochs=3, device=device)

    # Save the fine-tuned model
    trained_model.save_pretrained('path/to/save/fine_tuned_model')
    tokenizer.save_pretrained('path/to/save/fine_tuned_model')

if __name__ == "__main__":
    main()
