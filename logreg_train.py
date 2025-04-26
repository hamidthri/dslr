import sys
import csv
import numpy as np
import json

def load_dataset(filename):
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
    
    
def preprocess_data(headers, rows, houses):
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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    predictions = sigmoid(X @ theta)
    cost = (-1 / m) * (y @ np.log(predictions) + (1 - y) @ np.log(1 - predictions))
    return cost
    

def gradient_descent(X, y, theta, learning_rate=0.01, num_iterations=1000):
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
    
    for i in range(num_iterations):
        predictions = sigmoid(X @ theta)
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        theta -= learning_rate * gradient
        cost = compute_cost(X, y, theta)
        costs.append(cost)
        
    return theta, costs
           
        
def train_logestic_regression(X, y_encoded, unique_houses, learning_rate=0.01, num_iterations=1000):
    num_features = X.shape[1]
    trained_models = {}
    
    for house in unique_houses:
        y = y_encoded[house]
        theta = np.zeros(num_features)
        
        theta, costs = gradient_descent(X, y, theta, learning_rate, num_iterations)        
        trained_models[house] = theta.tolist()
        
    return trained_models
    


def save_model(model, feature_names, feature_means, feature_stds, unique_houses, output_file):
    model_data = {
        "feature_names": feature_names,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "houses": unique_houses,
        "weights": model
    }
    
    try:
        with open(output_file, 'w') as file:
            json.dump(model_data, file)
    except Exception as e:
        print(f"Error saving the model: {e}")

def main(train_file, output_file="model_weights.json"):
    headers, rows, houses = load_dataset(train_file)
    X_norm, y_encoded, feature_names, feature_means, feature_stds, unique_houses = preprocess_data(headers, rows, houses)
    
    # training
    print("Training the model...")
    trained_model = train_logestic_regression(X_norm, y_encoded, unique_houses, learning_rate=0.01, num_iterations=1000)
    
    # save_model(trained_model, feature_names, feature_means, feature_stds, unique_houses, output_file)
    print(f"Model saved to {output_file}")
    
    
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py <dataset_train.csv> [output_model.json]")
        sys.exit(1)
    
    train_file = sys.argv[1] if len(sys.argv) > 1 else "dataset_train.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "model_weights.json"
    
    main(train_file, output_file)
