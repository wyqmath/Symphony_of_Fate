import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
from midiutil import MIDIFile
from midi2audio import FluidSynth
import multiprocessing
from tqdm import tqdm
import traceback
import contextlib
import gc

# --- MAPPING DEFINITIONS ---
# These dictionaries map amino acids to musical properties.
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

# --- CORE FUNCTIONS ---

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
        # Suppress stdout/stderr to hide all library output
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
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
        plt.close(fig) # Crucial for preventing memory leaks in loops
        
        return True, None
    except Exception as e:
        return False, f"Spectrogram Error: {e}\n{traceback.format_exc()}"

# --- MULTIPROCESSING WORKER FUNCTION ---
def process_csv_row(args):
    """
    Worker function to process a single protein row from a CSV.
    This function is executed in a separate process.
    """
    protein_name, chain_seq, wav_output_dir, spectrogram_output_dir, soundfont_path = args
    
    temp_midi_dir = os.path.join(wav_output_dir, 'temp_midi')
    midi_file_path = os.path.join(temp_midi_dir, f"{protein_name}.mid")
    wav_file_path = os.path.join(wav_output_dir, f"{protein_name}.wav")

    try:
        os.makedirs(temp_midi_dir, exist_ok=True)
        spectrogram_path = os.path.join(spectrogram_output_dir, f"{protein_name}.png")
        
        chain_seq = chain_seq.upper().replace('*', '')
        if not chain_seq:
            return f"[SKIPPED] Protein '{protein_name}': Sequence is empty after cleaning."

        music_data = convert_to_music(chain_seq)
        
        if all(p == 'Rest' for p in music_data[0]):
            return f"[SKIPPED] Protein '{protein_name}': Sequence contains no mappable amino acids."

        success, error_msg = save_music_data_to_midi(music_data, midi_file_path, protein_name)
        if not success:
            return f"[FAILED] Protein '{protein_name}' at MIDI creation. Reason: {error_msg}"

        success, error_msg = convert_midi_to_wav(midi_file_path, wav_file_path, soundfont_path)
        if not success:
            return f"[FAILED] Protein '{protein_name}' at MIDI->WAV conversion. Reason: {error_msg}"

        if os.path.exists(wav_file_path):
            success, error_msg = generate_spectrogram(wav_file_path, spectrogram_path)
            if not success:
                return f"[FAILED] Protein '{protein_name}' at Spectrogram generation. Reason: {error_msg}"
        else:
            return f"[FAILED] Protein '{protein_name}': WAV file was not created."

        # On success, return None
        return None

    except Exception as e:
        return f"[UNEXPECTED ERROR] Protein '{protein_name}': {e}\n{traceback.format_exc()}"

    finally:
        # Clean up intermediate files regardless of success or failure
        if os.path.exists(midi_file_path):
            try: os.remove(midi_file_path)
            except OSError: pass
        # NOTE: The prompt asks to remove WAV files, but you might want to keep them.
        # If you want to keep the WAV files, comment out the next 3 lines.
        if os.path.exists(wav_file_path):
            try: os.remove(wav_file_path)
            except OSError: pass
        # Explicitly suggest garbage collection to help release memory
        gc.collect()

# --- MAIN FUNCTION (REFACTORED FOR MEMORY STABILITY) ---
def main():
    """
    Main function to set up paths, read CSVs, and run the pipeline.
    This version recreates the process pool for each batch to ensure
    stable memory usage over long runs.
    """
    # --- CONFIGURATION ---
    soundfont_path = "GeneralUser.sf2" # <-- IMPORTANT: CHANGE THIS TO YOUR .sf2 FILE PATH
    target_count_per_class = 5000      # The desired number of final images per class
    batch_size = 100                   # Process in chunks for efficiency and memory management
    TASK_TIMEOUT_SECONDS = 60          # Timeout for a single sequence processing in seconds

    input_csv_files = [f'ec_class_{i}.csv' for i in range(1, 8)]
    output_base_dir = 'enzyme_output'

    if not os.path.exists(soundfont_path):
        print(f"FATAL ERROR: SoundFont file not found at '{soundfont_path}'")
        return

    num_processes = os.cpu_count()
    all_error_messages = []

    print(f"Using {num_processes} processes for parallel execution.")
    print(f"Targeting {target_count_per_class} successful images per class.")
    print(f"Processing will be done in batches of {batch_size}, with a fresh process pool for each batch.")
    print(f"Setting a timeout of {TASK_TIMEOUT_SECONDS} seconds for each task.")

    for csv_filename in input_csv_files:
        if not os.path.exists(csv_filename):
            print(f"\n[WARNING] Input CSV file not found, skipping: {csv_filename}")
            continue

        file_base_name = os.path.splitext(csv_filename)[0]
        specific_wav_dir = os.path.join(output_base_dir, 'wav', file_base_name)
        specific_img_dir = os.path.join(output_base_dir, 'image', file_base_name)
        os.makedirs(specific_wav_dir, exist_ok=True)
        os.makedirs(specific_img_dir, exist_ok=True)
        
        print(f"\n--- Starting processing for file: {csv_filename} ---")
        
        try:
            df = pd.read_csv(csv_filename)
            df.columns = ['EC number', 'Sequence']
        except Exception as e:
            all_error_messages.append(f"[ERROR] Could not parse file '{csv_filename}'. Details: {e}")
            continue

        tasks_for_file = []
        for index, row in df.iterrows():
            sequence = str(row['Sequence'])
            ec_number = str(row['EC number'])
            if 'X' in sequence.upper():
                continue
            sanitized_name = f"{ec_number.replace('.', '_')}_{index}"
            task_args = (sanitized_name, sequence, specific_wav_dir, specific_img_dir, soundfont_path)
            tasks_for_file.append(task_args)

        if not tasks_for_file:
            print(f"No valid sequences found in {csv_filename}. Skipping.")
            continue

        successful_count = 0
        total_potential_tasks = len(tasks_for_file)
        
        # Process the full list of tasks in batches
        for i in range(0, total_potential_tasks, batch_size):
            if successful_count >= target_count_per_class:
                print(f"\nTarget of {target_count_per_class} reached for {csv_filename}. Moving to next file.")
                break

            batch_tasks = tasks_for_file[i:i + batch_size]
            
            # *** THE KEY CHANGE: The Pool is created and destroyed inside the loop ***
            with multiprocessing.Pool(processes=num_processes) as pool:
                desc = f"File {file_base_name} (Success: {successful_count}/{target_count_per_class})"
                pbar = tqdm(total=len(batch_tasks), desc=desc, position=0, leave=True)
                
                results = [pool.apply_async(process_csv_row, args=(task,)) for task in batch_tasks]

                for idx, future in enumerate(results):
                    try:
                        error = future.get(timeout=TASK_TIMEOUT_SECONDS)
                        if error:
                            all_error_messages.append(error)
                        else:
                            successful_count += 1
                    except multiprocessing.TimeoutError:
                        protein_name = batch_tasks[idx][0]
                        all_error_messages.append(f"[TIMEOUT] Protein '{protein_name}' exceeded {TASK_TIMEOUT_SECONDS}s limit.")
                    except Exception as e:
                        protein_name = batch_tasks[idx][0]
                        all_error_messages.append(f"[FUTURE ERROR] Protein '{protein_name}': {e}")
                    
                    pbar.set_description(f"File {file_base_name} (Success: {successful_count}/{target_count_per_class})")
                    pbar.update(1)
                
                pbar.close()

        if successful_count < target_count_per_class:
            print(f"\nFinished all {total_potential_tasks} potential sequences in {csv_filename}. Final count: {successful_count}/{target_count_per_class}.")

    print("\n--- All processing complete. ---")
    
    if all_error_messages:
        print("\n\n========================================")
        print("         SUMMARY OF ERRORS/SKIPS")
        print("========================================")
        print(f"A total of {len(all_error_messages)} processing failures or skips occurred.")
        
        # Save all errors to a log file
        error_log_path = os.path.join(output_base_dir, "error_log.txt")
        with open(error_log_path, 'w') as f:
            for msg in all_error_messages:
                f.write(f"{msg}\n")
        print(f"Full details have been saved to: {error_log_path}")

        # Show a sample in the console
        print("\nShowing the first 20 messages as examples:")
        for i, msg in enumerate(all_error_messages[:20]):
            print(f"\n{i+1}. {msg}")
        print("\n========================================")
    else:
        print("\nProcessing finished with no errors reported.")

if __name__ == "__main__":
    # This is necessary for multiprocessing to work correctly on some platforms (like Windows)
    multiprocessing.freeze_support()
    main()