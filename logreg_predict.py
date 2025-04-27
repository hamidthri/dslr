import sys
import csv
import numpy as np
import json

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
        
        for house in weights:
            weights[house] = np.array(weights[house])
            
        return weights, feature_names, feature_means, feature_stds, houses
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
            
            for row in reader:
                if index_col is not None:
                    indices.append(row[index_col])
                else:
                    indices.append(str(indices))
                    
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

def predict(X, weights, houses):
    predictions = []
    probs = {}
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

def save_predictions(indices, predictions, output_file="houses.csv"):
    """Save predictions to a CSV file."""
    try:
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Hogwarts House'])
            
            for idx, house in zip(indices, predictions):
                writer.writerow([idx, house])
        
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def main():
    weights, feature_names, feature_means, feature_stds, houses = load_model(model_file)
    header, rows, indices, feature_indices = load_test_data(test_file, feature_names)
    X_norm = preprocess_test_data(rows, feature_indices, feature_means, feature_stds)
    
    # make predictions
    predictions = predict(X_norm, weights, houses)
    
    # save predictions
    save_predictions(indices, predictions, output_file)
    

if  __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <model_file>")
        sys.exit(1)
        
    test_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "save/model_weights.json"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "predictions.json"
    main(test_file, model_file, output_file)