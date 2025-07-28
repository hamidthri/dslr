import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

def read_dataset(filename):
    """Read dataset and organize it into columns."""
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader) 
            
            data = {header: [] for header in headers}
            
            for row in reader:
                for i, value in enumerate(row):
                    try:
                        data[headers[i]].append(float(value) if value else None)
                    except ValueError:
                        data[headers[i]].append(value)
                        
            return headers, data
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

def is_numeric_column(column):
    """Check if a column contains numeric values."""
    for value in column:
        if value == '' or value is None:
            continue
        try:
            float(value)
        except (ValueError, TypeError):
            return False
    return True

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient between two arrays."""
    if len(x) != len(y) or len(x) == 0:
        return 0
    
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

def find_most_similar_features(headers, data):
    """Find the two most similar (correlated) numeric features."""
    numeric_columns = []
    non_numeric_columns = []
    
    # Identify non-numeric columns
    for header in headers:
        if not is_numeric_column(data[header]):
            non_numeric_columns.append(header)
    
    for header in headers:
        if header not in non_numeric_columns:
            if any(isinstance(val, (int, float)) for val in data[header] if val is not None):
                numeric_columns.append(header)
    
    if len(numeric_columns) < 2:
        print("Need at least 2 numeric columns to find similarity")
        return None, None, 0
    
    max_correlation = 0
    best_pair = (None, None)
    
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            feature1, feature2 = numeric_columns[i], numeric_columns[j]
            
            valid_pairs = [(data[feature1][k], data[feature2][k]) 
                          for k in range(len(data[feature1])) 
                          if data[feature1][k] is not None and data[feature2][k] is not None]
            
            if len(valid_pairs) < 2:
                continue
                
            x_vals, y_vals = zip(*valid_pairs)
            correlation = abs(calculate_correlation(x_vals, y_vals))
            
            if correlation > max_correlation:
                max_correlation = correlation
                best_pair = (feature1, feature2)
    
    return best_pair[0], best_pair[1], max_correlation

def create_scatter_plot(headers, data):
    """Create scatter plot of the two most similar features."""
    feature1, feature2, correlation = find_most_similar_features(headers, data)
    
    if feature1 is None or feature2 is None:
        print("Could not find similar features to plot")
        return
    
    print(f"Most similar features: {feature1} and {feature2}")
    print(f"Correlation coefficient: {correlation:.4f}")
    
    valid_indices = [k for k in range(len(data[feature1])) 
                    if data[feature1][k] is not None and data[feature2][k] is not None]
    
    has_house_data = 'Hogwarts House' in headers
    
    plt.figure(figsize=(10, 8))
    
    if has_house_data:
        color_map = {
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow',
            'Ravenclaw': 'blue',
            'Slytherin': 'green'
        }
        
        for house in set(data['Hogwarts House']):
            if house is not None:
                house_x = [data[feature1][k] for k in valid_indices 
                          if data['Hogwarts House'][k] == house]
                house_y = [data[feature2][k] for k in valid_indices 
                          if data['Hogwarts House'][k] == house]
                
                plt.scatter(house_x, house_y, alpha=0.7, s=50,
                           color=color_map.get(house, 'gray'), label=house)
        
        plt.legend()
    else:
        x_values = [data[feature1][k] for k in valid_indices]
        y_values = [data[feature2][k] for k in valid_indices]
        plt.scatter(x_values, y_values, alpha=0.7, s=50)
    
    plt.xlabel(feature1, fontsize=12)
    plt.ylabel(feature2, fontsize=12)
    plt.title(f'Scatter Plot: {feature1} vs {feature2}\n(Correlation: {correlation:.4f})', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if len(valid_indices) > 1:
        x_vals = [data[feature1][k] for k in valid_indices]
        y_vals = [data[feature2][k] for k in valid_indices]
        
        mean_x = sum(x_vals) / len(x_vals)
        mean_y = sum(y_vals) / len(y_vals)
        
        numerator = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(len(x_vals)))
        denominator = sum((x_vals[i] - mean_x) ** 2 for i in range(len(x_vals)))
        
        if denominator != 0:
            slope = numerator / denominator
            intercept = mean_y - slope * mean_x
            
            x_trend = [min(x_vals), max(x_vals)]
            y_trend = [slope * x + intercept for x in x_trend]
            plt.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2, label='Trend line')
    
    plt.tight_layout()
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
        sys.exit(1)
    
    headers, data = read_dataset(sys.argv[1])
    create_scatter_plot(headers, data)