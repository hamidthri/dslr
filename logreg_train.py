import sys
import csv

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
    pass


def main(train_file, output_file="model_weights.json"):
    headers, rows, houses = load_dataset(train_file)
    X_norm, y_encoded, feature_names, feature_means, feature_stds, unique_houses = preprocess_data(headers, rows, houses)
    
    
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py <dataset_train.csv> [output_model.json]")
        sys.exit(1)
    
    train_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "model_weights.json"
    
    main(train_file, output_file)
