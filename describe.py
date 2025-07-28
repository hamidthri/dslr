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
                        data[i].append(None)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    return headers, data

def compute_stats(values):
    # Remove None values manually
    cleaned = []
    for v in values:
        if v is not None:
            cleaned.append(v)

    # Manual count
    count = 0
    for _ in cleaned:
        count += 1

    if count == 0:
        return [0, 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']

    # Manual sum
    total = 0
    for v in cleaned:
        total += v

    mean = total / count

    # Manual min/max
    min_val = cleaned[0]
    max_val = cleaned[0]
    for v in cleaned:
        if v < min_val:
            min_val = v
        if v > max_val:
            max_val = v

    # Manual std (sample)
    variance_sum = 0
    for v in cleaned:
        variance_sum += (v - mean) ** 2
    std = (variance_sum / (count - 1)) ** 0.5 if count > 1 else 0

    # Manual sort (Selection Sort)
    for i in range(count):
        min_index = i
        for j in range(i + 1, count):
            if cleaned[j] < cleaned[min_index]:
                min_index = j
        # Swap
        cleaned[i], cleaned[min_index] = cleaned[min_index], cleaned[i]

    # Manual percentile
    def percentile(p):
        k = (count - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < count else f
        fraction = k - f
        return cleaned[f] + (cleaned[c] - cleaned[f]) * fraction

    return [
        count,
        mean,
        std,
        min_val,
        percentile(25),
        percentile(50),
        percentile(75),
        max_val
    ]



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
    headers, columns = read_csv(file_path)

    # Collect numeric columns only (ignore first few like names, etc.)
    print(f"{'Feature':<30} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'Max':>10}")
    for i, col in enumerate(columns):
        numeric_values = [v for v in col if isinstance(v, float)]
        if numeric_values:  # skip categorical columns
            stats = compute_stats(col)
            print(f"{headers[i]:<30} {stats[0]:8.2f} {stats[1]:10.2f} {stats[2]:10.2f} {stats[3]:10.2f} {stats[4]:10.2f} {stats[5]:10.2f} {stats[6]:10.2f} {stats[7]:10.2f}")    numeric_headers = []
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
