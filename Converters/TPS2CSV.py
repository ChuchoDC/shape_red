'''
This function transforms your TPS files into CSV files
'''

def tps2csv(tps_file, csv_file, num_landmarks):
    import csv
    data = []
    current_entry = None
    
    with open(tps_file, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("IMAGE="):
                if current_entry is not None:
                    data.append(current_entry)
                image_name = line.split("=")[1]
                current_entry = [image_name]
            elif line and not line.startswith(("LM=", "ID=")):
                if current_entry is not None:
                    x, y = map(float, line.split())
                    current_entry.extend([x, y])
        
        if current_entry is not None:
            data.append(current_entry)
    
    if len(data) > 1:
        names = [entry[0] for entry in data[1:]]
        
        all_coords = []
        for entry in data[:-1]:
            if len(entry) > 1:  
                all_coords.append(entry[1:])

        min_length = min(len(names), len(all_coords))
        corrected_data = [[names[i]] + all_coords[i] for i in range(min_length)]
    else:
        corrected_data = []

    headers = ["id"] + [f"{coord}{i}" for i in range(num_landmarks) for coord in ["X", "Y"]]

    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(corrected_data)
    
    print(f'Archivo CSV generado correctamente: {csv_file}')

# Example:
# tps2csv('InputFile.tps', 'OutputFile.csv', #Landmarks)