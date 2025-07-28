import csv
import matplotlib.pyplot as plt

def read_csv(path, selected_headers):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        indices = [headers.index(h) for h in selected_headers]
        data = [[] for _ in selected_headers]

        for row in reader:
            for idx, col_index in enumerate(indices):
                try:
                    data[idx].append(float(row[col_index]))
                except ValueError:
                    data[idx].append(None)
    return selected_headers, data

def clean(col_x, col_y):
    return [
        (x, y) for x, y in zip(col_x, col_y)
        if x is not None and y is not None
    ]

def plot_matrix(headers, data):
    n = len(headers)
    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                ax.hist([x for x in data[i] if x is not None], bins=15, alpha=0.7)
                ax.set_title(headers[i])
            else:
                points = clean(data[j], data[i])
                if points:
                    xs, ys = zip(*points)
                    ax.scatter(xs, ys, alpha=0.3, s=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("pair_plot.png")
    plt.show()

if __name__ == "__main__":
    selected = [
        "Astronomy",
        "Defense Against the Dark Arts",
        "Herbology",
        "Charms",
        "Transfiguration"
    ]
    headers, data = read_csv("dataset_train.csv", selected)
    plot_matrix(headers, data)
