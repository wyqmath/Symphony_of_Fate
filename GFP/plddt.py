import matplotlib.pyplot as plt
import pandas as pd
import os

def extract_plddt_from_pdb(pdb_file):
    """
    Parses a PDB file to extract pLDDT scores from the B-factor column
    for each CA (alpha-carbon) atom.

    Args:
        pdb_file (str): The path to the PDB file.

    Returns:
        list: A list of pLDDT scores (float). Returns an empty list if file not found.
    """
    if not os.path.exists(pdb_file):
        print(f"Warning: File not found at {pdb_file}. Skipping.")
        return []

    plddt_scores = []
    with open(pdb_file, 'r') as f:
        for line in f:
            # We only care about ATOM records for alpha-carbons (CA)
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                try:
                    # The B-factor is in columns 61-66
                    b_factor_str = line[60:66].strip()
                    plddt_scores.append(float(b_factor_str))
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse B-factor on line in {pdb_file}: {line.strip()}")
    
    return plddt_scores

def main():
    """
    Main function to process PDB files, create a comparative plot,
    and save the data to a CSV file.
    """
    # --- Configuration ---
    # Dictionary mapping labels for the plot to their PDB filenames
    pdb_files = {
        'WT Baseline (2wur)': '2wur.pdb',
        'Generated 1': 'gen1.pdb',
        'Generated 2': 'gen2.pdb',
        'Generated 3': 'gen3.pdb'
    }
    
    output_plot_file = 'plddt_comparison_plot.png'
    output_csv_file = 'plddt_scores.csv'

    # --- Data Extraction ---
    all_plddt_data = {}
    print("Starting pLDDT extraction...")
    for label, filename in pdb_files.items():
        scores = extract_plddt_from_pdb(filename)
        if scores: # Only add if scores were successfully extracted
            all_plddt_data[label] = scores
            print(f"  - Extracted {len(scores)} residues from {filename}")

    if not all_plddt_data:
        print("Error: No data was extracted. Please check your PDB file names and paths.")
        return

    # --- Plotting ---
    print(f"\nGenerating plot and saving to {output_plot_file}...")
    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    for label, scores in all_plddt_data.items():
        # The x-axis is the residue number for each specific protein
        residue_numbers = range(1, len(scores) + 1)
        ax.plot(residue_numbers, scores, label=f"{label} ({len(scores)} residues)", lw=2)

    # Formatting the plot
    ax.set_title('Comparative pLDDT Profile of Generated GFP Variants vs. Wild-Type', fontsize=16)
    ax.set_xlabel('Residue Number', fontsize=12)
    ax.set_ylabel('pLDDT Score (Confidence)', fontsize=12)
    ax.set_ylim(0, 1.05) # pLDDT is on a 0-1 scale in your files
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Save the figure with high resolution
    plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
    print("Plot saved successfully.")
    # plt.show() # Uncomment this line if you want the plot to pop up when you run the script

    # --- Save Data to CSV ---
    # Because residue counts differ, pandas will handle this by filling shorter
    # columns with 'NaN' (Not a Number), which is the correct behavior.
    print(f"\nSaving data to {output_csv_file}...")
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_plddt_data.items()]))
    df.index = df.index + 1 # Make the index start from 1 (Residue 1)
    df.index.name = 'Residue_Number'
    df.to_csv(output_csv_file)
    print("CSV file saved successfully.")


if __name__ == '__main__':
    main()