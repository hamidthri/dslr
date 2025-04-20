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
    values = [v for v in values if v is not None]
    if not values:
        return [0, 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']

    count = len(values)
    mean = sum(values) / count
    sorted_vals = sorted(values)

    def percentile(p):
        k = (count - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, count - 1)
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

    std = (sum((x - mean) ** 2 for x in values) / (count - 1)) ** 0.5 if count > 1 else 0
    return [
        count,
        mean,
        std,
        min(values),
        percentile(25),
        percentile(50),
        percentile(75),
        max(values)
    ]

def describe(file_path):
    headers, columns = read_csv(file_path)

    # Collect numeric columns only (ignore first few like names, etc.)
    print(f"{'Feature':<30} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'Max':>10}")
    for i, col in enumerate(columns):
        numeric_values = [v for v in col if isinstance(v, float)]
        if numeric_values:  # skip categorical columns
            stats = compute_stats(col)
            print(f"{headers[i]:<30} {stats[0]:8.2f} {stats[1]:10.2f} {stats[2]:10.2f} {stats[3]:10.2f} {stats[4]:10.2f} {stats[5]:10.2f} {stats[6]:10.2f} {stats[7]:10.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    describe(sys.argv[1])
