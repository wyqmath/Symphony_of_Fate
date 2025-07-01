import os
import multiprocessing
from Bio import SeqIO
from tqdm import tqdm

# --- MAPPING DEFINITION ---
# This set is created from your original mapping keys. It defines all valid characters.
VALID_AMINO_ACIDS = set('ARNDCEQGHILKMFPSTWYV')

def find_invalid_characters(sequence):
    """
    Checks a sequence for any characters not in the VALID_AMINO_ACIDS set.
    
    Args:
        sequence (str): The amino acid sequence string.
        
    Returns:
        list: A list of formatted strings, each describing an invalid character and its position.
              Returns an empty list if the sequence is valid.
    """
    invalid_chars_found = []
    # We iterate through the sequence with an index (starting from 1 for biological convention)
    for i, amino_acid in enumerate(sequence.upper(), 1):
        if amino_acid not in VALID_AMINO_ACIDS:
            invalid_chars_found.append(f"'{amino_acid}' at position {i}")
    return invalid_chars_found

def validate_fasta_record(args):
    """
    Worker function for multiprocessing. It processes a single protein record.
    
    Args:
        args (tuple): A tuple containing the record object, the original filename,
                      and the set of valid amino acids.
                      
    Returns:
        str: An error message if invalid characters are found.
        None: If the sequence is valid.
    """
    record, filename = args
    
    # Get the sequence from the record
    chain_seq = str(record.seq)

    if not chain_seq:
        # Sanitize the record ID for clear reporting
        sanitized_id = record.id.replace('|', '_').replace(':', '_').replace('/', '\\') if record.id else "N/A"
        return f"[WARNING] Protein '{sanitized_id}' in file '{filename}' has an empty sequence."

    # Find all invalid characters in the sequence
    invalid_chars = find_invalid_characters(chain_seq)
    
    if invalid_chars:
        # If any invalid characters were found, format a detailed report string
        sanitized_id = record.id.replace('|', '_').replace(':', '_').replace('/', '\\') if record.id else "N/A"
        details = ", ".join(invalid_chars)
        return f"[INVALID] Protein '{sanitized_id}' in file '{filename}': Found {details}."
        
    return None # Return None if the sequence is valid

def main():
    """
    Main function to set up paths, run the validation pipeline, and report all findings.
    """
    # --- CONFIGURATION ---
    # Define the directories containing your FASTA files.
    # You can add more paths here, like the 'test' directory.
    input_dirs = [
        'fasta/train',
        # 'fasta/test'
    ]
    
    # Use a recommended number of processes for parallel execution
    num_processes = max(1, os.cpu_count() - 1)
    print(f"Using {num_processes} processes for parallel validation.")
    
    all_report_messages = []
    tasks = []

    # --- Prepare all tasks from all specified directories ---
    print("\n--- Preparing tasks for validation ---")
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"Warning: Input directory not found, skipping: {input_dir}")
            continue
            
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.fasta', '.fa')):
                input_fasta_path = os.path.join(input_dir, filename)
                try:
                    # Parse all records from the FASTA file
                    records = list(SeqIO.parse(input_fasta_path, "fasta"))
                    if not records:
                        all_report_messages.append(f"[WARNING] FASTA file '{filename}' is empty or contains no valid records.")
                        continue
                    
                    # Add each record as a task for the worker pool
                    for record in records:
                        tasks.append((record, filename))
                except Exception as e:
                    all_report_messages.append(f"[ERROR] Could not parse file '{filename}'. It may be corrupted. Details: {e}")

    if not tasks:
        print("\nNo valid protein sequences found to process.")
        return
        
    # --- Process all tasks in parallel ---
    print(f"\n--- Validating {len(tasks)} total sequences ---")
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a progress bar to monitor the validation process
        pbar = tqdm(pool.imap_unordered(validate_fasta_record, tasks), total=len(tasks), desc="Validating sequences")
        # Collect the results (which are report messages or None)
        for report in pbar:
            if report:
                all_report_messages.append(report)
            
    print("\n--- All validation complete. ---")
    
    # --- FINAL REPORT ---
    if all_report_messages:
        print("\n\n========================================")
        print("         VALIDATION REPORT")
        print("========================================")
        # Sort messages for cleaner output (optional, but nice)
        all_report_messages.sort()
        for i, msg in enumerate(all_report_messages, 1):
            print(f"{i}. {msg}")
        print("\n========================================")
    else:
        print("\nSuccess! All sequences were validated with no issues found.")

if __name__ == "__main__":
    # This is necessary for multiprocessing to work correctly on some OSes (like Windows)
    multiprocessing.freeze_support()
    main()