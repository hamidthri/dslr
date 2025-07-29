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

def create_pair_plot(headers, data):
    """Create a pair plot (scatter plot matrix) for numeric features."""
    numeric_columns = []
    non_numeric_columns = ['Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    
    for header in headers:
        if header not in non_numeric_columns:
            if any(isinstance(val, (int, float)) for val in data[header] if val is not None):
                numeric_columns.append(header)
    
    if len(numeric_columns) > 5:
        print(f"Limiting pair plot to the first 5 numeric features out of {len(numeric_columns)}")
        numeric_columns = numeric_columns[:5]

    n = len(numeric_columns)
    if n == 0:
        print("No numeric columns found for pair plot")
        return
    
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    has_house_data = 'Hogwarts House' in headers
    
    color_map = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    
    for i, feature1 in enumerate(numeric_columns):
        for j, feature2 in enumerate(numeric_columns):
            ax = axes[i, j] if n > 1 else axes
            
            # On diagonal, plot histogram
            if i == j:
                # Filter out None values
                valid_values = [val for val in data[feature1] if val is not None]
                
                if has_house_data:
                    # Group by house for histograms
                    for house in set(data['Hogwarts House']):
                        if house is not None:
                            house_values = [data[feature1][k] for k in range(len(data[feature1])) 
                                           if data['Hogwarts House'][k] == house 
                                           and data[feature1][k] is not None]
                            ax.hist(house_values, bins=20, alpha=0.5,
                                   color=color_map.get(house, None), label=house)
                else:
                    ax.hist(valid_values, bins=20)
                
                ax.set_title(feature1, fontsize=8)
            else:
                valid_indices = [k for k in range(len(data[feature1])) 
                               if data[feature1][k] is not None and data[feature2][k] is not None]
                
                if has_house_data:
                    # Group by house for scatter plots
                    for house in set(data['Hogwarts House']):
                        if house is not None:
                            house_x = [data[feature1][k] for k in valid_indices 
                                     if data['Hogwarts House'][k] == house]
                            house_y = [data[feature2][k] for k in valid_indices 
                                     if data['Hogwarts House'][k] == house]
                            ax.scatter(house_x, house_y, s=5, alpha=0.7,
                                      color=color_map.get(house, None))
                else:
                    x_values = [data[feature1][k] for k in valid_indices]
                    y_values = [data[feature2][k] for k in valid_indices]
                    ax.scatter(x_values, y_values, s=5, alpha=0.7)
                
                ax.set_xlabel(feature1, fontsize=8)
                ax.set_ylabel(feature2, fontsize=8)
            
            # Make tick labels smaller
            ax.tick_params(axis='both', labelsize=6)
    
    # Add a legend for house colors (only once)
    if has_house_data:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=house, markersize=8) 
                  for house, color in color_map.items() 
                  if house in set(data['Hogwarts House'])]
        fig.legend(handles=handles, loc='upper right')
    
    plt.suptitle('Pair Plot of Numeric Features', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Leave space for the title
    plt.show()
    plt.savefig('pair_plot.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <dataset.csv>")
        sys.exit(1)
    
    headers, data = read_dataset(sys.argv[1])
    create_pair_plot(headers, data)