import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from Bio import SeqIO
from midiutil import MIDIFile
from midi2audio import FluidSynth
import multiprocessing
from tqdm import tqdm
import traceback # To get detailed error information

# --- MAPPING DEFINITIONS (Unchanged) ---
map_rd_scale = {
    'A': 'C4', 'R': 'D4', 'N': 'E4', 'D': 'F4', 'C': 'G4', 'E': 'A4', 'Q': 'B4',
    'G': 'C5', 'H': 'D5', 'I': 'E5', 'L': 'F5', 'K': 'G5', 'M': 'A5', 'F': 'B5',
    'P': 'C6', 'S': 'D6', 'T': 'E6', 'W': 'F6', 'Y': 'G6', 'V': 'A6'
}
map_rd_ium = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8, 'H': 9,
    'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17,
    'W': 18, 'Y': 19, 'V': 20
}
map_rd_rhythm = {
    'A': 72, 'R': 144, 'N': 216, 'D': 288, 'C': 360, 'E': 432, 'Q': 504,
    'G': 576, 'H': 648, 'I': 720, 'L': 792, 'K': 864, 'M': 936, 'F': 1008,
    'P': 1080, 'S': 1152, 'T': 1224, 'W': 1296, 'Y': 1368, 'V': 1440
}

# --- CORE FUNCTIONS (Unchanged) ---

def convert_to_music(chain_seq):
    """Converts an amino acid sequence into musical data."""
    pitches = [map_rd_scale.get(aa, 'Rest') for aa in chain_seq]
    timbres = [map_rd_ium.get(aa, 0) for aa in chain_seq]
    rhythms = [map_rd_rhythm.get(aa, 480) for aa in chain_seq]
    return pitches, timbres, rhythms

def save_music_data_to_midi(music_data, output_path, protein_name):
    """Returns (True, None) on success, or (False, error_message) on failure."""
    try:
        midi = MIDIFile(1)
        track, time = 0, 0
        midi.addTrackName(track, time, protein_name)
        midi.addTempo(track, time, 120)
        pitches, _, rhythms = music_data
        for pitch, rhythm in zip(pitches, rhythms):
            if pitch != 'Rest':
                note_char, octave = pitch[0], int(pitch[1])
                note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                note = 12 * octave + note_map.get(note_char, 0)
                duration = rhythm / 480.0
                midi.addNote(track, 0, note, time, duration, 100)
                time += duration
        with open(output_path, 'wb') as output_file:
            midi.writeFile(output_file)
        return True, None
    except Exception as e:
        return False, f"MIDI Write Error: {e}\n{traceback.format_exc()}"

def convert_midi_to_wav(midi_file_path, wav_file_path, soundfont_path, sample_rate=16000):
    """Returns (True, None) on success, or (False, error_message) on failure."""
    try:
        fs = FluidSynth(sound_font=soundfont_path, sample_rate=sample_rate)
        fs.midi_to_audio(midi_file_path, wav_file_path)
        return True, None
    except Exception as e:
        return False, f"FluidSynth Error: {e}\n{traceback.format_exc()}"

def generate_spectrogram(wav_file_path, output_image_path):
    """
    Generates a spectrogram with a blue colormap and white background.
    Returns (True, None) on success, or (False, error_message) on failure.
    """
    try:
        y, sr = librosa.load(wav_file_path, sr=None)
        if np.max(np.abs(y)) == 0:
            return False, "WAV file is silent, cannot generate spectrogram."
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig = plt.figure(figsize=(10, 4), facecolor='white')
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('white')

        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='Blues', ax=ax)
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close(fig)
        
        return True, None
    except Exception as e:
        return False, f"Spectrogram Error: {e}\n{traceback.format_exc()}"

# --- MULTIPROCESSING WORKER FUNCTION (Unchanged) ---
def process_fasta_record(args):
    """
    Worker function to process a single protein record.
    Returns an error string on failure, or None on success.
    """
    record, protein_name, wav_output_dir, spectrogram_output_dir, soundfont_path = args
    
    chain_seq = str(record.seq).upper().replace('*', '')

    if not chain_seq:
        return f"[SKIPPED] Protein '{protein_name}': Sequence is empty after cleaning."

    temp_midi_dir = os.path.join(wav_output_dir, 'temp_midi')
    os.makedirs(temp_midi_dir, exist_ok=True)
    midi_file_path = os.path.join(temp_midi_dir, f"{protein_name}.mid")
    wav_file_path = os.path.join(wav_output_dir, f"{protein_name}.wav")
    spectrogram_path = os.path.join(spectrogram_output_dir, f"{protein_name}.png")

    # --- Pipeline Execution ---
    music_data = convert_to_music(chain_seq)
    
    success, error_msg = save_music_data_to_midi(music_data, midi_file_path, protein_name)
    if not success:
        return f"[FAILED] Protein '{protein_name}' at MIDI creation. Reason: {error_msg}"

    success, error_msg = convert_midi_to_wav(midi_file_path, wav_file_path, soundfont_path)
    if not success:
        if os.path.exists(midi_file_path): os.remove(midi_file_path)
        return f"[FAILED] Protein '{protein_name}' at MIDI->WAV conversion. Reason: {error_msg}"

    try:
        if os.path.exists(wav_file_path):
            success, error_msg = generate_spectrogram(wav_file_path, spectrogram_path)
            if not success:
                return f"[FAILED] Protein '{protein_name}' at Spectrogram generation. Reason: {error_msg}"
    finally:
        # Cleanup intermediate files
        if os.path.exists(midi_file_path):
            try: os.remove(midi_file_path)
            except OSError: pass
        if os.path.exists(wav_file_path):
            try: os.remove(wav_file_path)
            except OSError: pass
            
    return None # Return None on success

# --- MAIN FUNCTION (CORRECTED) ---
def main():
    """
    Main function to set up paths, run the pipeline, and report errors at the end.
    """
    # --- CONFIGURATION ---
    soundfont_path = "GeneralUser.sf2" # <-- CHANGE THIS TO YOUR .sf2 FILE PATH
    
    # --- Set the maximum number of sequences to process PER FILE ---
    processing_limit = 200

    if not os.path.exists(soundfont_path):
        print(f"FATAL ERROR: SoundFont file not found at '{soundfont_path}'")
        return

    path_configs = [
        #{'input': 'fasta/train', 'wav_out': 'train_wav', 'img_out': 'train_image'},
         {'input': 'fasta/test',  'wav_out': 'test_wav',  'img_out': 'test_image'}
    ]
    
    num_processes = os.cpu_count()
    all_error_messages = []
    tasks = []

    print(f"Using {num_processes} processes for parallel execution.")
    print(f"Searching for up to {processing_limit} valid sequences per file to process...")

    for config in path_configs:
        input_dir = config['input']
        base_wav_dir = config['wav_out']
        base_img_dir = config['img_out']
        
        if not os.path.isdir(input_dir):
            print(f"Warning: Input directory not found, skipping: {input_dir}")
            continue
            
        os.makedirs(base_wav_dir, exist_ok=True)
        os.makedirs(base_img_dir, exist_ok=True)
        
        print(f"\n--- Preparing tasks from directory: {input_dir} ---")
        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(('.fasta', '.fa')):
                continue

            input_fasta_path = os.path.join(input_dir, filename)
            file_base_name = os.path.splitext(filename)[0]
            specific_wav_dir = os.path.join(base_wav_dir, file_base_name)
            specific_img_dir = os.path.join(base_img_dir, file_base_name)
            os.makedirs(specific_wav_dir, exist_ok=True)
            os.makedirs(specific_img_dir, exist_ok=True)

            try:
                records = list(SeqIO.parse(input_fasta_path, "fasta"))
                if not records:
                    all_error_messages.append(f"[WARNING] FASTA file '{filename}' is empty or contains no valid records.")
                    continue
                
                # --- LOGIC CORRECTION START ---
                # Counter for valid sequences found in THIS file.
                valid_sequences_found_in_file = 0
                
                # Loop through ALL records in the file to find valid ones.
                for record in records:
                    # First, check if we have already found enough valid sequences for this file.
                    if valid_sequences_found_in_file >= processing_limit:
                        break # Exit the loop for this file and move to the next.

                    # Second, check if the current sequence is valid (no 'X').
                    if 'X' in str(record.seq).upper():
                        continue # Skip this invalid record and check the next one.

                    # If we are here, the sequence is valid and we haven't hit the limit for this file yet.
                    # Add the task.
                    sanitized_id = record.id.replace('|', '_').replace(':', '_').replace('/', '\\') if record.id else f"{file_base_name}_seq{valid_sequences_found_in_file + 1}"
                    task_args = (record, sanitized_id, specific_wav_dir, specific_img_dir, soundfont_path)
                    tasks.append(task_args)
                    
                    # Finally, increment the counter for valid sequences found.
                    valid_sequences_found_in_file += 1
                # --- LOGIC CORRECTION END ---

            except Exception as e:
                all_error_messages.append(f"[ERROR] Could not parse file '{filename}'. It may be corrupted. Skipping. Details: {e}")

    if not tasks:
        print("\nNo valid protein sequences found to process.")
    else:
        print(f"\n--- Collected a total of {len(tasks)} valid sequences. Starting processing. ---")
        with multiprocessing.Pool(processes=num_processes) as pool:
            pbar = tqdm(pool.imap_unordered(process_fasta_record, tasks), total=len(tasks), desc="Processing sequences")
            for error in pbar:
                if error:
                    all_error_messages.append(error)
            
    print("\n--- All processing complete. ---")
    
    # --- FINAL ERROR REPORT ---
    if all_error_messages:
        print("\n\n========================================")
        print("         SUMMARY OF ERRORS/SKIPS")
        print("========================================")
        for i, msg in enumerate(all_error_messages):
            print(f"\n{i+1}. {msg}")
        print("\n========================================")
    else:
        print("\nProcessing finished with no errors reported.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()