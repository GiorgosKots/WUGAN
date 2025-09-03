def calculate_percentages(file_path, type="random forest"):
    # Initialize counters
    total = 0
    above_0_5 = 0
    above_0_8 = 0

    # Open and read the file
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        for line in file:
            # Split each line into parts
            parts = line.strip().split('\t')
            if len(parts) == 3:
                # Extract the probability and convert it to a float
                probability = float(parts[2])
                total += 1

                # Check if the probability is above 0.5 or 0.8
                if probability > 0.5:
                    above_0_5 += 1
                if probability > 0.8:
                    above_0_8 += 1

    # Calculate the percentages
    percentage_above_0_5 = (above_0_5 / total) * 100
    percentage_above_0_8 = (above_0_8 / total) * 100

    # Print the results
    print(f"Percentage above 0.5 ({type}): {percentage_above_0_5:.2f}%")
    print(f"Percentage above 0.8 ({type}): {percentage_above_0_8:.2f}%")

    return percentage_above_0_5, percentage_above_0_8

def calculate_averages(file_paths, types):
    """
    Calculate percentages for multiple files and their averages
    
    Args:
        file_paths (list): List of file paths to process
        types (list): List of model types corresponding to each file
    """
    if len(file_paths) != len(types):
        raise ValueError("Number of file paths must match number of types")

    total_0_5 = 0
    total_0_8 = 0
    
    # Process each file
    for file_path, model_type in zip(file_paths, types):
        try:
            above_0_5, above_0_8 = calculate_percentages(file_path, model_type)
            total_0_5 += above_0_5
            total_0_8 += above_0_8
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return

    # Calculate and print averages
    avg_0_5 = total_0_5 / len(file_paths)
    avg_0_8 = total_0_8 / len(file_paths)
    
    print("\nAverages across all models:")
    print(f"Average percentage above 0.5: {avg_0_5:.2f}%")
    print(f"Average percentage above 0.8: {avg_0_8:.2f}%")

# Example usage:
# calculate_percentages("D:\GANs\GANs\campr4\CAMPdownload_2025-05-21 22-28-41.txt")