from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import torch
from tqdm import tqdm

# Function to train Naïve Bayes Model
def train_and_evaluate_nb(vectorizer, X_train, X_test, y_train, y_test, filename, vectorizer_name):
    # Vectorize the data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize Naïve Bayes classifier
    nb_classifier = MultinomialNB()
    
    # Train the classifier
    nb_classifier.fit(X_train_vec, y_train)
    
    # Predictions on test set
    y_pred = nb_classifier.predict(X_test_vec)

    # Extract misclassified examples
    misclassified_indices = y_test != y_pred
    misclassified_text = X_test[misclassified_indices]
    true_labels = y_test[misclassified_indices]
    predicted_labels = y_pred[misclassified_indices]
    
    # Create list of dictionaries for misclassified examples
    misclassified_data = []
    for text, true_label, predicted_label in zip(misclassified_text, true_labels, predicted_labels):
        misclassified_data.append({
            'Text': text,
            'True Label': int(true_label),  # Convert to int if necessary
            'Predicted Label': int(predicted_label),  # Convert to int if necessary
            'Vectorizer': vectorizer_name,
            'File': filename
        })
    

    print(f"Vectorizer: {vectorizer_name}\n")
    # Print classification report
    print("\nClassification Report:")
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Count misclassifications
    misclassified_count = misclassified_indices.sum()
    total_examples = len(y_test)
    print(f"Number of misclassified examples out of {total_examples} examples : {misclassified_count}")
    print(50*"*")
    
    return classification_report_dict, misclassified_data, misclassified_count, total_examples


# Function for training and evaluating FFNN model
def train_and_evaluate_ffnn(model, train_loader, test_loader, criterion, optimizer, device):
    # Training
    model.to(device).train()
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.to(device).eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().detach().numpy())
            true_labels.extend(labels.cpu().detach().numpy())
    
    predictions = [1 if p >= 0.5 else 0 for p in predictions]
    

    f1 = f1_score(y_pred=predictions, y_true=true_labels)
    print(f"F1: {f1}")
    # Classification report
    print(classification_report(true_labels, predictions))
    # Count misclassified examples
    misclassified_count = sum([1 for true, pred in zip(true_labels, predictions) if true != pred])
    total_examples = len(true_labels)
    
    print(f"\nNumber of misclassified examples out of {total_examples} examples : {misclassified_count}")
    print(50*"*")


def load_json_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def prepare_train_test_split(data, m, test_size=0.2, stratify_col='category', random_state=42):
    """Prepare train and test sets."""
    df = pd.DataFrame(data)
    if m == "nb":
        X = df[['preprocessed_text']]  # Features
    else:
        X = df['preprocessed_text']  # Features

    y = df[stratify_col]           # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, stratify=y,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

# Move samples from train set to test set
def move_samples_between_sets(X_train, y_train, X_test, y_test, category_to_move=0, n_samples=750, random_state=42):
    """Move samples of a specific category from the train set to the test set."""
    # Identify indices of the specified category to move to the test set
    category_indices = y_train[y_train == category_to_move].index
    # Randomly select indices to move
    moved_indices = resample(category_indices, replace=False, n_samples=n_samples, random_state=random_state)
    # Move samples from train set to test set
    X_test = pd.concat([X_test, X_train.loc[moved_indices]])
    y_test = pd.concat([y_test, y_train.loc[moved_indices]])
    # Remove samples from train set
    X_train = X_train.drop(index=moved_indices)
    y_train = y_train.drop(index=moved_indices)
    return X_train, y_train, X_test, y_test