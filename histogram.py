import sys
import csv
import numpy as np


def read_dataset(filename):
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            house_index = None
            house_column_names = ['Hogwarts House', 'House']
            
            for name in house_column_names:
                if name in headers:
                    house_index = headers.index(name)
                    break
            
            if house_index is None:
                print("Error: Hogwarts House column not found")
                sys.exit(1)
            
            houses_data = {}
            
            for row in reader:
                if len(row) <= house_index:
                    continue
                    
                house = row[house_index]
                if not house:  
                    continue
                
                if house not in houses_data:
                    houses_data[house] = {header: [] for header in headers}
                
                for i, value in enumerate(row):
                    try:
                        houses_data[house][headers[i]].append(float(value) if value else None)
                    except ValueError:
                        houses_data[house][headers[i]].append(value)
                        
            return headers, houses_data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)


def find_homogeneous_courses(headers, houses_data, non_course_columns=None):
    # Default non-course columns if not provided
    if non_course_columns is None:
        non_course_columns = [
            'Index', 'Hogwarts House', 'House', 'First Name', 'Last Name',
            'Birthday', 'Best Hand', 'Birth', 'Name'
        ]
    
    course_columns = []
    
    for header in headers:
        if header not in non_course_columns:
            has_numeric = False
            for house_data in houses_data.values():
                sample_values = house_data[header]
                if any(isinstance(val, (int, float)) for val in sample_values if val is not None):
                    has_numeric = True
                    break
                    
            if has_numeric:
                course_columns.append(header)
    
    homogeneity_scores = {}
    
    for course in course_columns:
        course_data = {}
        for house, data in houses_data.items():
            course_data[house] = [val for val in data[course] if val is not None]
        
        if any(len(data) < 5 for data in course_data.values()):
            continue
        
        bin_range = (
            min(min(data) for data in course_data.values()),
            max(max(data) for data in course_data.values())
        )
        total_points = sum(len(data) for data in course_data.values())
        num_bins = min(20, max(10, int(np.sqrt(total_points / len(course_data)))))
        bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
        
        hist_counts = {}
        for house, data in course_data.items():
            counts, _ = np.histogram(data, bins=bins)
            if sum(counts) > 0:
                hist_counts[house] = counts / sum(counts)
            else:
                hist_counts[house] = counts
        
        bin_stds = []
        for bin_idx in range(num_bins):
            bin_values = [hist_counts[house][bin_idx] for house in course_data.keys()]
            bin_stds.append(np.std(bin_values))
        
        homogeneity_scores[course] = np.mean(bin_stds)
    
    sorted_courses = sorted(homogeneity_scores.items(), key=lambda x: x[1])
    
    return sorted_courses, course_data
                

def plot_histograms_for_top_courses(headers, houses_data):
    # courses sorted by the homogeneity of their values
    sorted_courses, _ = find_homogeneous_courses(headers, houses_data)
    if not sorted_courses:
        print("No homogeneous courses found.")
        return


def main():
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <file_path>")
        return
    file_path = sys.argv[1]
    
    headers, houses_data = read_dataset(file_path)




if __name__ == "__main__":
    main()