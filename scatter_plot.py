import csv
import matplotlib.pyplot as plt

def read_numeric_data(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        columns = [[] for _ in headers]

        for row in reader:
            for i, value in enumerate(row):
                try:
                    columns[i].append(float(value))
                except ValueError:
                    columns[i].append(None)
                    
        numeric_headers = []
        numeric_columns = []
        for h, col in zip(headers, columns):
            if all(x is None or isinstance(x, float) for x in col):
                numeric_headers.append(h)
                numeric_columns.append(col)
        return numeric_headers, numeric_columns

def mean(col):
    vals = [v for v in col if v is not None]
    return sum(vals) / len(vals)

def std(col, m):
    vals = [v for v in col if v is not None]
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

def correlation(x, y):
    x_vals = [a for a, b in zip(x, y) if a is not None and b is not None]
    y_vals = [b for a, b in zip(x, y) if a is not None and b is not None]
    if len(x_vals) == 0:
        return 0
    mx, my = mean(x_vals), mean(y_vals)
    sx, sy = std(x_vals, mx), std(y_vals, my)
    cov = sum((a - mx) * (b - my) for a, b in zip(x_vals, y_vals)) / len(x_vals)
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0

def find_most_correlated(headers, data):
    best = 0
    pair = (None, None)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            r = abs(correlation(data[i], data[j]))
            if r > best:
                best = r
                pair = (i, j)
    return pair

if __name__ == "__main__":
    headers, data = read_numeric_data("dataset_train.csv")
    i, j = find_most_correlated(headers, data)

    x, y = data[i], data[j]
    x_vals = [a for a, b in zip(x, y) if a is not None and b is not None]
    y_vals = [b for a, b in zip(x, y) if a is not None and b is not None]

    plt.scatter(x_vals, y_vals, alpha=0.5)
    plt.xlabel(headers[i])
    plt.ylabel(headers[j])
    plt.title(f"{headers[i]} vs {headers[j]}")
    plt.tight_layout()
    plt.savefig("scatter_plot.png")
    plt.show()
