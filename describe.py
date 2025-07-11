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


def calculate_min(column):
    min_value = None
    for value in column:
        if value == '' or value is None:
            continue
        if min_value is None or value < min_value:
            min_value = value
    return min_value

def calculate_max(column):
    max_value = None
    for value in column:
        if value == '' or value is None:
            continue
        if max_value is None or value > max_value:
            max_value = value
    return max_value

def calculate_percentile(column, percentile):
    valid_values = [value for value in column if value != '' and value is not None]
    if not valid_values:
        return None
    
    sorted_values = sorted(valid_values)
    n = len(sorted_values)
    
    index = (n - 1) * (percentile / 100)
    
    if index.is_integer():
        return sorted_values[int(index)]
    
    lower_index = int(index)
    upper_index = lower_index + 1
    
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    
    fraction = index - lower_index
    
    return lower_value + (upper_value - lower_value) * fraction 

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
            elif stat == "Min": 
                value = calculate_min(column)
            elif stat == "25%":
                value = calculate_percentile(column, 25)
            elif stat == "50%":
                value = calculate_percentile(column, 50)
            elif stat == "75%":
                value = calculate_percentile(column, 75)
            elif stat == "Max":
                value = calculate_max(column)
                
            print(f"{value:15.2f}", end="")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    describe(sys.argv[1])


## we need to use the train data and then for further we need test data only to predict the howards since its NUll
