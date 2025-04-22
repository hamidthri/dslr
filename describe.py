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

