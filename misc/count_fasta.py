import os

def count_sequences_in_fasta(file_path):
    """
    Counts the number of sequences in a given FASTA file.

    A sequence is identified by a header line starting with '>'.

    Args:
        file_path (str): The path to the FASTA file.

    Returns:
        int: The total number of sequences found, or None if an error occurs.
    """
    # First, check if the file even exists to avoid the error message.
    if not os.path.exists(file_path):
        # We can print a softer message here since we expect some files might be missing.
        print(f"Info: File '{file_path}' not found, skipping.")
        return None

    sequence_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Check if the line is a header line
                if line.strip().startswith('>'):
                    sequence_count += 1
    except Exception as e:
        print(f"An error occurred while processing '{file_path}': {e}")
        return None
        
    return sequence_count

# This is the main part of the script that will run.
if __name__ == "__main__":
    print("--- Starting FASTA sequence count for files 1.fasta to 9.fasta ---")
    
    # We will loop through numbers 1 to 9
    for i in range(1, 10):
        # Create the filename for the current iteration (e.g., "1.fasta")
        filename = f"{i}.fasta"
        
        # Call our function to count the sequences
        count = count_sequences_in_fasta(filename)
        
        # If the function returned a valid count (i.e., not None), print the result.
        if count is not None:
            print(f"Result: The file '{filename}' contains {count} sequences.")
            
    print("\n--- Script finished ---")