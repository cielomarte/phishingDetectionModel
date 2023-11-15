#defines the model, sets up training loop, saving trained model

#import needed libraries
from tqdm import tqdm
import torch
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup #-- using Adam optimizer with weight decay fix and get_linear_schedule_with_warmup as learning rate scheduler
from torch.utils.data import DataLoader #data loader is a utility to load data in batches

#initialize model -- initializes BERT model for sequence classification
# uses num_labels for classification - binary
# loads pre-trained BERT model and configures it for sequence classification with specified number of labels
def initialize_model(num_labels=2):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model


#set up training loop
def train_model(model, train_dataset, val_dataset, epochs=20, batch_size=8, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    tqdm_ep = list(range(epochs))
    for epoch in tqdm(tqdm_ep):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            model.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        tot_pred_c = 0
        tot_pred_count = len(val_loader)
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs, labels=labels)
            total_val_loss += outputs.loss.item()
            label_preds = outputs.logits.argmax(axis=1)
            tot_pred_c += (label_preds == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {tot_pred_c / tot_pred_count:.4f}")

    return model




#save trained model
def save_model(model, save_path="model_save"):
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


