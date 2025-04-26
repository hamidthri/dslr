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


def load_test_data(test_file):
    pass

def preprocess_test_data(test_data):
    pass

def main():
    weights, feature_names, feature_means, feature_stds, houses = load_model(model_file)
    _ = load_test_data(test_file)
    preprocess_test_data(_)
    

if  __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <model_file>")
        sys.exit(1)
        
    test_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "save/model_weights.json"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "predictions.json"
    main(test_file, model_file, output_file)