
#PREprocessing ------------
from data_preprocessing import load_data


# set debugging options here
# _SUBSET_DATA = 10000
_SUBSET_DATA = None


# Define directory and filenames
directory = 'C:\\Users\\17136\\PycharmProjects\\phishingDetectionModel'
zip_filename = 'phishingLegURLSKaggleDataSet.zip'
csv_filename = 'new_data_urls.csv'  # This can be omitted if you want to use the first CSV in the ZIP

# Load the data -----------------------------
data = load_data(directory, zip_filename, csv_filename)
print(data.head())  # Print the first few rows to inspect the data
#----------------------------TEST-----------------------------#
print("Loading data finished successfully...")

# call functions to preprocess data
from data_preprocessing import load_data, handle_missing_values, tokenize_data, split_data, create_datasets

# Load the data
#data = load_data('path_to_dataset.csv')

if _SUBSET_DATA is not None:
    data = data.iloc[:_SUBSET_DATA]

# Handle missing values
data = handle_missing_values(data)
#----------------------------TEST-----------------------------#
print("Missing values handled successfully....")

# Tokenize the data
tokenized_data = tokenize_data(data, 'url')
#----------------------------TEST-----------------------------#
print("Data tokenized successfully....")

# Split the data
train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(tokenized_data, data['status'])
#----------------------------TEST-----------------------------#
print("Data split successfully....")

# Create PyTorch datasets
train_dataset, val_dataset, test_dataset = create_datasets(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
#----------------------------TEST-----------------------------#
print("PyTorch datasets created successfully....")

#train the model ---------------------------
from model_training import initialize_model, train_model, save_model

# Initialize the model
model = initialize_model()

# Train the model
trained_model = train_model(model, train_dataset, val_dataset, epochs=20, batch_size=8)
#----------------------------TEST-----------------------------#
print("Model trained successfully....")



#----------------------------TEST-----------------------------#
import os
print("The working directory is ..")
print(os.getcwd())
# Save the trained model
save_model(trained_model, "./phishing_detection_model")
#----------------------------TEST-----------------------------#
print("Trained model saved successfully....")


#model evaluation --------------------------------------------
from model_evaluation import load_trained_model, evaluate_model, calculate_metrics

# Load the trained model
model = load_trained_model("phishing_detection_model")

# Evaluate the model on the test dataset
preds, labels = evaluate_model(model, test_dataset)
#----------------------------TEST-----------------------------#
print("Model evaluated successfully ....")


# Calculate and print metrics
calculate_metrics(preds, labels)

