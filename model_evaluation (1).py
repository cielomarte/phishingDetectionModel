#evaluates trained model non test set
#calculates metrics like accuracy, precision, recall etc

#import needed libraries
import torch
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score


#load the new trained model
def load_trained_model(model_path="phishing_detection_model"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model


#evaluate model
def evaluate_model(model, test_dataset, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []

    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels



#calculate metrics
def calculate_metrics(preds, labels):
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=["Legitimate", "Phishing"]))
    print("Accuracy:", accuracy_score(labels, preds))




