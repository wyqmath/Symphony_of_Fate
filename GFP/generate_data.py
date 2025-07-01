import pandas as pd
import sys
import re # Import the regular expressions module

def apply_mutations(wild_type_seq, mutation_str):
    """
    Applies one or more mutations to a wild-type sequence using robust regex parsing.
    Handles mutations separated by colons, with extraneous prefixes and stop codons.
    
    Args:
        wild_type_seq (str): The original protein sequence.
        mutation_str (str): A string describing the mutations, e.g., "SA108D" or "SA108D:SN144D".
        
    Returns:
        str: The mutated sequence.
    """
    if not isinstance(mutation_str, str) or not mutation_str.strip():
        return wild_type_seq

    mutable_seq = list(wild_type_seq)
    
    # This is the regex pattern to find a valid mutation.
    # It looks for:
    # ([A-Z])   - A single letter (the original amino acid)
    # (\d+)     - One or more digits (the position)
    # ([A-Z\*]) - A single letter OR a '*' (the new amino acid or stop codon)
    # $         - Anchors the search to the END of the string.
    mutation_pattern = re.compile(r'([A-Z])(\d+)([A-Z\*])$')
    
    mutations = mutation_str.split(':')
    
    for mut in mutations:
        mut = mut.strip()
        if not mut:
            continue
        
        match = mutation_pattern.search(mut)
        
        if match:
            original_aa = match.group(1)
            position = int(match.group(2))
            new_aa = match.group(3)
            
            # Convert 1-indexed position to 0-indexed for Python
            idx = position - 1
            
            if 0 <= idx < len(mutable_seq):
                # We can add a check to see if the original AA matches, which is good practice
                if mutable_seq[idx] == original_aa:
                    mutable_seq[idx] = new_aa
                else:
                    # This warning is useful for spotting potential issues in the source data
                    # print(f"Warning: Mismatch at position {position} for mutation '{mut}'. Expected {mutable_seq[idx]}, but mutation says {original_aa}. Applying anyway.")
                    mutable_seq[idx] = new_aa
            else:
                print(f"Warning: Position {position} in mutation '{mut}' is out of bounds. Skipping.")
        else:
            # This will catch any truly malformed mutation strings that the regex can't parse.
            print(f"Warning: Could not parse mutation format for '{mut}'. Skipping.")
            
    return "".join(mutable_seq)

# --- Main execution ---

wild_type_avGFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"

input_filename = 'amino_acid_genotypes_to_brightness.tsv'
output_filename = 'GFP.csv'

print(f"Attempting to process '{input_filename}' with robust parser...")

try:
    df = pd.read_csv(input_filename, sep='\t')
    
    if 'aaMutations' not in df.columns:
        print(f"Error: Column 'aaMutations' not found. Available columns are: {df.columns.tolist()}")
        sys.exit()

    df['Sequence'] = df['aaMutations'].fillna('').apply(lambda m: apply_mutations(wild_type_avGFP, m))
    
    final_df = df[['Sequence']].copy()
    final_df['conv(%)'] = df['medianBrightness']
    
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nSuccess! Processed data has been saved to '{output_filename}'.")
    print("This version should have handled the formatting errors.")
    print("Here is a preview of the output file:")
    print(final_df.head())

except FileNotFoundError:
    print(f"Error: The input file '{input_filename}' was not found.")
    print("Please make sure it is in the same directory as this script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")