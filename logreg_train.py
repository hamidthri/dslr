import sys
import csv
import numpy as np
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import pandas as pd


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


def load_dataset(filename: str) -> Tuple[List[str], List[List[str]], List[str]]:
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            house_index = headers.index('Hogwarts House') if 'Hogwarts House' in headers else None
            if house_index is None:
                print("Error: Hogwarts House column not found")
                sys.exit(1)
            
            rows = []
            houses = []
            for row in reader:
                if len(row)  != len(headers) or not row[house_index]:
                    continue
                houses.append(row[house_index])
                rows.append(row)
            
            return headers, rows, houses
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)
    
    
def preprocess_data(headers: List[str], rows: List[List[str]], houses: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str], List[float], List[float], List[str]]:
    non_feature_cols = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    feature_indices = []
    feature_names = []
    for i, header in enumerate(headers):
        if header not in non_feature_cols:
            feature_indices.append(i)
            feature_names.append(header)
    
    X = []
    for row in rows:
        features = []
        for idx in feature_indices:
            try:
                val = float(row[idx])
                features.append(val)
            except ValueError:
                features.append(0.0)
        X.append(features)
    
    # convert to numpy array
    X = np.array(X)
    # check for NaN values and replace them with mean of the column
    for col in range(X.shape[1]):
        col_mean = np.nanmean(X[:, col])
        nan_indices = np.isnan(X[:, col])
        X[nan_indices, col] = col_mean
        
    # normalize the data using z-score normalization
    X_norm = np.zeros_like(X)    
    feature_means = []
    feature_stds = []
    
    for col in range(X.shape[1]):
        col_mean = np.mean(X[:, col])
        col_std = np.std(X[:, col])
        std = col_std if col_std != 0 else 1
        
        X_norm[:, col] = (X[:, col] - col_mean) / std
        feature_means.append(col_mean)
        feature_stds.append(col_std)
        
    X_norm = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))    
    unique_houses = sorted(set(houses))
    
    y_encoded = {}
    for house in unique_houses:
        y_encoded[house] = np.array([1 if h == house else 0 for h in houses])
        
    return X_norm, y_encoded, feature_names, feature_means, feature_stds, unique_houses

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = len(y)
    predictions = sigmoid(X @ theta)
    cost = (-1 / m) * (y @ np.log(predictions) + (1 - y) @ np.log(1 - predictions))
    return cost
    

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray,
                     learning_rate: float, num_iterations: int, live_plot: bool = False,
                     house_name: Optional[str] = None) -> Tuple[np.ndarray, List[float]]:

    """
        dl / dz = dl / dyhat * dyhat / dz
        
        dl / dyhat = -(y / yhat - (1 - y) / (1 - yhat))
        = -(y * (1 - yhat) - (1 - y) * yhat) / (yhat * (1 - yhat))
        = -(y - yhat) / (yhat * (1 - yhat))
        
        in the other hand 
        dyhat / dz = yhat * (1 - yhat)
        
        dl / dz = -(y - yhat) / (yhat * (1 - yhat)) * yhat * (1 - yhat) = yhat - y
        
        dl / dw = dl / dz * dz / dw = (yhat - y) * x
    """
    
    m = len(y)
    costs = []

    if live_plot:
        # plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], label=house_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title("Training Loss")
        ax.legend()
        xs, ys = [], []

    for i in range(num_iterations):
        predictions = sigmoid(X @ theta)
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        theta -= learning_rate * gradient
        cost = compute_cost(X, y, theta)
        costs.append(cost)

        if i % 100 == 0 and live_plot:
            logging.info(f"Iteration {i}: Cost = {cost}")
            xs.append(i)
            ys.append(cost)
            line.set_xdata(xs)
            line.set_ydata(ys)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    if live_plot:
        plt.ioff()
        plt.show()

    return theta, costs

def stochastic_gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray,
                               learning_rate: float, num_iterations: int, batch_size: int = 1,
                               live_plot: bool = False, house_name: Optional[str] = None) -> Tuple[np.ndarray, List[float]]:
    m = len(y)
    costs = []
    indices = np.arange(m)

    if live_plot:
        fig, ax = plt.subplots()
        line, = ax.plot([], [], label=house_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title("Training Loss (SGD)")
        ax.legend()
        xs, ys = [], []

    for i in range(num_iterations):
        if i % (m // batch_size) == 0:
            np.random.shuffle(indices)
            
        for j in range(0, m, batch_size):
            batch_indices = indices[j:j+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            predictions = sigmoid(X_batch @ theta)
            errors = predictions - y_batch
            gradient = (1 / len(batch_indices)) * (X_batch.T @ errors)
            
            theta -= learning_rate * gradient
        
        cost = compute_cost(X, y, theta)
        costs.append(cost)
        
        if i % 50 == 0 and live_plot:
            logging.info(f"SGD Iteration {i}: Cost = {cost}")
            xs.append(i)
            ys.append(cost)
            line.set_xdata(xs)
            line.set_ydata(ys)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    if live_plot:
        plt.ioff()
        plt.show()

    return theta, costs
           
        
def train_logestic_regression(X: np.ndarray, y_encoded: Dict[str, np.ndarray], unique_houses: List[str],
                             learning_rate: float=0.01, num_iterations: int=1000, 
                             method: str='batch', batch_size: int=32) -> Dict[str, List[float]]:
    num_features = X.shape[1]
    trained_models = {}
    
    for house in unique_houses:
        y = y_encoded[house]
        theta = np.zeros(num_features)
        logging.info(f"Training for house: {house} using method: {method}")
        
        if method.lower() == 'sgd':
            theta, costs = stochastic_gradient_descent(X, y, theta, learning_rate, num_iterations, 
                                                      batch_size=batch_size, live_plot=True, house_name=house)
        else:
            theta, costs = gradient_descent(X, y, theta, learning_rate, num_iterations, 
                                           live_plot=True, house_name=house)
            
        print(f"Final cost for {house}: {costs[-1]}")    
        trained_models[house] = theta.tolist()
        
    return trained_models
    


def save_model(model: Dict[str, List[float]], feature_names: List[str], feature_means: List[float],
               feature_stds: List[float], unique_houses: List[str], output_file: str, method: str = "batch") -> None:
    model_data = {
        "feature_names": feature_names,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "houses": unique_houses,
        "weights": model,
        "training_method": method
    }
    
    try:
        with open(output_file, 'w') as file:
            json.dump(model_data, file, indent=4)
    except Exception as e:
        print(f"Error saving the model: {e}")


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


def confusion_matrix_analysis(trained_model: Dict[str, List[float]], X_norm: np.ndarray,
                              houses: List[str], unique_houses: List[str]) -> np.ndarray:
    confusion_matrix = np.zeros((len(unique_houses), len(unique_houses)), dtype=int)

    house_to_idx = {house: idx for idx, house in enumerate(unique_houses)}

    for k in range(len(X_norm)):
        probs = {}
        for house in unique_houses:
            probs[house] = sigmoid(np.dot(X_norm[k], np.array(trained_model[house])))

        predicted_house = max(probs, key=probs.get)
        true_house = houses[k]

        true_idx = house_to_idx[true_house]
        pred_idx = house_to_idx[predicted_house]

        confusion_matrix[true_idx, pred_idx] += 1
    df_conf = pd.DataFrame(confusion_matrix, index=unique_houses, columns=unique_houses)
    print("\nConfusion Matrix:")
    print(df_conf)
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: list) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(confusion_matrix, cmap='Blues')
    plt.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='left')
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = confusion_matrix[i, j]
            ax.text(j, i, str(value), va='center', ha='center', color='black' if value < confusion_matrix.max()/2 else 'white')

    plt.tight_layout()
    plt.show()

def print_metrics_table(metrics: Dict[str, Dict[str, float]]):
    df = pd.DataFrame(metrics).T
    df = df[["precision", "recall", "f1_score"]]  # ensure column order
    df.columns = ["Precision", "Recall", "F1 Score"]
    print(df.round(3))


def main(train_file: str, output_file: str="save/model_weights.json", method: str="batch", batch_size: int=32) -> None:
    headers, rows, houses = load_dataset(train_file)
    X_norm, y_encoded, feature_names, feature_means, feature_stds, unique_houses = preprocess_data(headers, rows, houses)
    
    print(f"Training the model using {method} method...")
    trained_model = train_logestic_regression(X_norm, y_encoded, unique_houses, 
                                             learning_rate=0.01, num_iterations=1000,
                                             method=method, batch_size=batch_size)
    
    confusion_matrix = confusion_matrix_analysis(trained_model, X_norm, houses, unique_houses)
    plot_confusion_matrix(confusion_matrix, unique_houses)
    
    print("\nMetrics from confusion matrix:")
    metrics = evaluate_metrics_from_confusion(confusion_matrix, unique_houses)

    print_metrics_table(metrics)

    
    
    save_model(trained_model, feature_names, feature_means, feature_stds, unique_houses, output_file, method)
    print(f"Model saved to {output_file}")
    
    
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train logistic regression model for Hogwarts houses prediction')
    parser.add_argument('train_file', nargs='?', default="datasets/dataset_train.csv", 
                        help='Path to training dataset CSV file')
    parser.add_argument('output_file', nargs='?', default="save/model_weights.json", 
                        help='Path to save the model weights JSON file')
    parser.add_argument('-m', '--method', choices=['batch', 'sgd'], default='batch', 
                        help='Training method: batch (default) or sgd')
    parser.add_argument('-b', '--batch_size', type=int, default=32, 
                        help='Batch size for SGD (default: 32, ignored if method is batch)')
    
    args = parser.parse_args()
    
    main(args.train_file, args.output_file, args.method, args.batch_size)