import sys
import csv
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional

def load_model(model_file):
    try:
        with open(model_file, 'r') as file:
            model = json.load(file)
            print(f"Model loaded from {model_file}")
        weights = model["weights"]
        feature_names = model["feature_names"]
        feature_means = model["feature_means"]
        feature_stds = model["feature_stds"]
        houses = model["houses"]
        # Extract the training method if available, otherwise default to "batch"
        training_method = model.get("training_method", "batch")
        print(f"Training method used: {training_method}")
        
        for house in weights:
            weights[house] = np.array(weights[house])
        return weights, feature_names, feature_means, feature_stds, houses, training_method
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)


def load_test_data(test_file, feature_names):
    try:
        with open(test_file, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            feature_indices = {}
            for feature in feature_names:
                if feature in headers:
                    feature_indices[feature] = headers.index(feature)
                
            index_col = headers.index("index") if "index" in headers else None
            
            rows = []
            indices = []
            
            for i, row in enumerate(reader):
                if index_col is not None:
                    indices.append(row[index_col])
                else:
                    indices.append(str(i))
                rows.append(row)

                
            return headers, rows, indices, feature_indices
    except Exception as e:
        print(f"Error loading the test data: {e}")
        sys.exit(1)
            
            

def preprocess_test_data(rows, feature_indices, feature_means, feature_stds):
    X = np.zeros((len(rows), len(feature_indices)))
    for i, row in enumerate(rows):
        for j, (feature, idx) in enumerate(feature_indices.items()):
            try:
                if idx < len(row) and row[idx]:
                    X[i, j] = float(row[idx])
                else:
                    X[i, j] = 0
            except ValueError:
                X[i, j] = 0
                
    for col in range(X.shape[1]):
        non_indices = np.isnan(X[:, col])
        X[non_indices, col] = feature_means[col]
        
    X_norm = np.zeros_like(X)
    for col in range(X.shape[1]):
        X_norm[:, col] = (X[:, col] - feature_means[col]) / feature_stds[col]
        
    # add bias
    X_norm = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
    return X_norm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights, houses, training_method="batch"):
    predictions = []
    probs = {}
    
    print(f"Making predictions using model trained with {training_method} method")
    
    for house in houses:
        probs[house] = sigmoid(np.dot(X, weights[house]))

    for i in range(X.shape[0]):
        max_prob = -1
        predicted_house = None
        for house in houses:
            if probs[house][i] > max_prob:
                max_prob = probs[house][i]
                predicted_house = house
        predictions.append(predicted_house)
    
    return predictions

import os

def save_predictions(indices, predictions, output_file="houses.csv"):
    """Save predictions to a CSV file."""
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Hogwarts House'])
            
            for idx, house in zip(indices, predictions):
                writer.writerow([idx, house])
        
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")


def test_confusion_matrix_analysis(predictions, true_labels, class_names):
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for true, pred in zip(true_labels, predictions):
        if true in class_to_idx and pred in class_to_idx:
            true_idx = class_to_idx[true]
            pred_idx = class_to_idx[pred]
            confusion_matrix[true_idx, pred_idx] += 1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: list, training_method: str = "batch") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(confusion_matrix, cmap='Blues')
    plt.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='left')
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix (Training Method: {training_method})")

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = confusion_matrix[i, j]
            ax.text(j, i, str(value), va='center', ha='center', color='black' if value < confusion_matrix.max()/2 else 'white')

    plt.tight_layout()
    plt.show()

def print_metrics_table(metrics: Dict[str, Dict[str, float]], training_method: str = "batch"):
    df = pd.DataFrame(metrics).T
    df = df[["precision", "recall", "f1_score"]]  # ensure column order
    df.columns = ["Precision", "Recall", "F1 Score"]
    print(f"\nMetrics (Training Method: {training_method}):")
    print(df.round(3))
    
def evaluate_metrics_from_confusion(confusion_matrix: np.ndarray, class_names: list) -> Dict[str, Dict[str, float]]:
    
    num_classes = confusion_matrix.shape[0]
    metrics = {}

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return metrics

def main(test_file, model_file, output_file):
    weights, feature_names, feature_means, feature_stds, houses, training_method = load_model(model_file)
    header, rows, indices, feature_indices = load_test_data(test_file, feature_names)
    X_norm = preprocess_test_data(rows, feature_indices, feature_means, feature_stds)

    predictions = predict(X_norm, weights, houses, training_method)

    # Save predictions to CSV
    save_predictions(indices, predictions, output_file)

    # Get true labels from test set
    house_index = header.index("Hogwarts House") if "Hogwarts House" in header else None

    if house_index is not None:
        true_labels = [row[house_index].strip() for row in rows if len(row) > house_index and row[house_index].strip()]
        if true_labels:
            cm = test_confusion_matrix_analysis(predictions, true_labels, houses)
            plot_confusion_matrix(cm, houses, training_method)
            metrics = evaluate_metrics_from_confusion(cm, houses)
            print_metrics_table(metrics, training_method)
        else:
            print("No true labels present in test file — skipping evaluation.")
    else:
        print("Hogwarts House column not found — skipping evaluation.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions using trained logistic regression model')
    parser.add_argument('test_file', help='Path to test dataset CSV file')
    parser.add_argument('-m', '--model', default="save/model_weights.json", 
                       help='Path to the model file (default: save/model_weights.json)')
    parser.add_argument('-o', '--output', default="prediction/houses.csv", 
                       help='Path to save predictions CSV (default: prediction/houses.csv)')
    
    args = parser.parse_args()
    
    main(args.test_file, args.model, args.output)