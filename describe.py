import sys
import csv


def read_csv(file_path):
    data = []
    headers = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            for _ in headers:
                data.append([])
            for row in reader:
                for i, value in enumerate(row):
                    try:
                        data[i].append(float(value))
                    except ValueError:
                        data[i].append(value)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    return headers, data


def is_numeric_column(column):
    for value in column:
        if value == '' or value is None:
            continue
        try:
            float(value)
        except (ValueError, TypeError):
            return False
    return True

def describe(file_path):
    headers, data = read_csv(file_path)
    
    numeric_headers = []
    numeric_data = []
    for i, column in enumerate(data):
        if is_numeric_column(column):
            numeric_headers.append(headers[i])
            numeric_data.append(column)
    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)
        
    describe(sys.argv[1])