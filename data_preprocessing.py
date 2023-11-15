# handles dataset , loading, inspecting, handling missing values, tokenization, splitting data into training,validation,test sets

# load dataset -------------------- #
import pandas as pd
import os
import zipfile
import torch


def load_data(directory, zip_filename, csv_filename=None):
    """
    Load data from a ZIP file containing a CSV.

    Parameters:
    - directory: Path to the directory containing the ZIP file.
    - zip_filename: Name of the ZIP file.
    - csv_filename: Name of the CSV file inside the ZIP. If not provided, the first CSV in the ZIP will be used.

    Returns:
    - DataFrame containing the data.
    """

    zip_filepath = os.path.join(directory, zip_filename)  # creates full path to zip, opens zip file, searches for csv file, then read into a pandas DataFrame

    with zipfile.ZipFile(zip_filepath, 'r') as z:
        # If csv_filename is not provided, find the first CSV in the ZIP
        if csv_filename is None:
            csv_filename = next((name for name in z.namelist() if name.endswith('.csv')), None)
            if csv_filename is not None:
                pass
            else:
                raise ValueError("No CSV found in the ZIP file.")

        with z.open(csv_filename) as f:
            data = pd.read_csv(f)

    return data


# handle missing values -----------#
# takes DataFrame and removes any rows with missing values via dropna method ---#
def handle_missing_values(data):
    data.dropna(inplace=True)
    return data


# tokenize ---#
#tokenizes the data using BERT tokenizer #
from transformers import BertTokenizer
#data = DataFrame with the data --#
#column_name = name of the column in the DataFrame that has the text data to be tokenized--#
def tokenize_data(data, column_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #initializes BERT tokenizer ---#
    return tokenizer(list(data[column_name]), padding=True, truncation=True, return_tensors="pt") #tokenizes data in specified column; tokenized data is returned in PyTorch sensor format ---#

#splits tokenized data into training, validation, and test sets
from sklearn.model_selection import train_test_split

def split_data(tokenized_data, labels):
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(tokenized_data['input_ids'], labels, test_size=0.3)  #uses train_test_split from scikit-learn to first split data into training set (70%) amd temp set (30%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5)
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels



#convert to pytorch dataset ---#
from torch.utils.data import TensorDataset

def create_datasets(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels): #create_datasets() converts the split data into PyTorch datasets ; params are the split data and labels ---#

    # Print shapes for debugging
    print("Shapes of train_texts:", train_texts.shape)
    print("Shapes of train_labels:", train_labels.shape)
    print("Shapes of val_texts:", val_texts.shape)
    print("Shapes of val_labels:", val_labels.shape)
    print("Shapes of test_texts:", test_texts.shape)
    print("Shapes of test_labels:", test_labels.shape)

    train_labels = torch.tensor(train_labels.values)
    val_labels = torch.tensor(val_labels.values)
    test_labels = torch.tensor(test_labels.values)

    #continue to create dataset---#
    train_dataset = TensorDataset(train_texts, train_labels)                #creates PyTorch datasets using TensorDataset for each training set
    val_dataset = TensorDataset(val_texts, val_labels)                      #creates PyTorch datasets using TensorDataset for each validation set
    test_dataset = TensorDataset(test_texts, test_labels)                   #creates PyTorch datasets using TensorDataset for each test set
    return train_dataset, val_dataset, test_dataset


