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

    # Count misclassifications
    misclassified = (y_test != y_pred).sum()
    total = len(y_test)
    print(f"\nFile: {filename} | Vectorizer: {vectorizer_name}")
    print(f"For {vectorizer}:\nNumber of misclassified examples out of {total} examples : {misclassified}")
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    return classification_report(y_test, y_pred, output_dict=True)

# Function for training and evaluating FFNN model
def train_and_evaluate_ffnn(model, train_loader, test_loader, criterion, optimizer, device):
    # Training
    model.train()
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().detach().numpy())
            true_labels.extend(labels.cpu().detach().numpy())
    
    predictions = [1 if p >= 0.5 else 0 for p in predictions]
    f1 = f1_score(true_labels, predictions)
    print(f"\nF1 Score: {f1}")
    
    # Classification report
    print(classification_report(true_labels, predictions))
    
    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))

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
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

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