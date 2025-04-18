import sys
import csv


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


def main():
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <file_path>")
        return
    file_path = sys.argv[1]
    
    headers, houses_data = read_dataset(file_path)




if __name__ == "__main__":
    main()