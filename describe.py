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



def calculate_count(column):
    count = 0
    for value in column:
        if value == '' or value is None:
            continue
        count += 1
    return count

def calculate_mean(column):
    total = 0
    count = 0
    for value in column:
        if value == '' or value is None:
            continue
        total += value
        count += 1
    return total / count if count > 0 else 0

def calculate_std(column):
    mean = calculate_mean(column)
    variance = 0
    count = 0
    for value in column:
        if value == '' or value is None:
            continue
        variance += (value - mean) ** 2
        count += 1
    return (variance / count) ** 0.5 if count > 0 else 0


def describe(file_path):
    headers, data = read_csv(file_path)
    
    numeric_headers = []
    numeric_data = []
    for i, column in enumerate(data):
        if is_numeric_column(column):
            numeric_headers.append(headers[i])
            numeric_data.append(column)
            
    statistics = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    # print the header
    print(f"{'':25}", end="")
    for header in numeric_headers:
        print(f"{header:12}", end="")
        
    print()
    
    for stat in statistics:
        print(f"{stat:15}", end="")
        for column in numeric_data:
            if stat == "Count":
                value = calculate_count(column)
            elif stat == "Mean":
                value = calculate_mean(column)
            elif stat == "Std": 
                value = calculate_std(column)
                
            print(f"{value:15.2f}", end="")
        print()
    
    
    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)
        
    describe(sys.argv[1])