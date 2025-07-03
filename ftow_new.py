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
    Creates a spectrogram and then deletes intermediate MIDI and WAV files.
    Returns an error string on failure, or None on success.
    """
    record, protein_name, base_output_dir, spectrogram_output_dir, soundfont_path = args
  
    # Define paths for intermediate and final files
    # A temporary directory for MIDI files within the main output folder
    temp_midi_dir = os.path.join(base_output_dir, 'temp_midi')
    os.makedirs(temp_midi_dir, exist_ok=True)
    
    midi_file_path = os.path.join(temp_midi_dir, f"{protein_name}.mid")
    wav_file_path = os.path.join(base_output_dir, f"{protein_name}.wav") # Temp WAV path
    spectrogram_path = os.path.join(spectrogram_output_dir, f"{protein_name}.png")

    try:
        chain_seq = str(record.seq).upper().replace('*', '')
        if not chain_seq:
            return f"[SKIPPED] Protein '{protein_name}': Sequence is empty after cleaning."

        # --- Pipeline Execution ---
        music_data = convert_to_music(chain_seq)
      
        success, error_msg = save_music_data_to_midi(music_data, midi_file_path, protein_name)
        if not success:
            return f"[FAILED] Protein '{protein_name}' at MIDI creation. Reason: {error_msg}"

        success, error_msg = convert_midi_to_wav(midi_file_path, wav_file_path, soundfont_path)
        if not success:
            return f"[FAILED] Protein '{protein_name}' at MIDI->WAV conversion. Reason: {error_msg}"

        if not os.path.exists(wav_file_path) or os.path.getsize(wav_file_path) == 0:
            return f"[FAILED] Protein '{protein_name}': WAV file was not created or is empty. FluidSynth may have failed silently."

        success, error_msg = generate_spectrogram(wav_file_path, spectrogram_path)
        if not success:
            # If spectrogram fails, we keep the WAV file for debugging but report the error.
            return f"[FAILED] Protein '{protein_name}' at Spectrogram generation. Reason: {error_msg}"
        
        # --- WAV FILE DELETION ON SUCCESS ---
        # If we get here, the spectrogram was created successfully.
        # Now, delete the intermediate WAV file to save disk space.
        try:
            os.remove(wav_file_path)
        except OSError:
            pass
            
    except Exception as e:
        return f"[FATAL WORKER ERROR] Protein '{protein_name}': {e}\n{traceback.format_exc()}"
        
    finally:
        # --- FINAL CLEANUP ---
        # Always try to remove the intermediate MIDI file, regardless of success or failure.
        if os.path.exists(midi_file_path):
            try:
                os.remove(midi_file_path)
            except OSError:
                pass
          
    return None # Return None on success

# --- MAIN FUNCTION (MODIFIED WITH TIMEOUTS) ---
def main():
    """
    Main function to set up paths, run the pipeline with timeouts, 
    and report errors at the end.
    """
    # --- CONFIGURATION ---
    soundfont_path = "GeneralUser.sf2" # <-- CHANGE THIS TO YOUR .sf2 FILE PATH
    processing_limit = 800 # Max number of valid sequences to process PER FASTA FILE
    TASK_TIMEOUT_SECONDS = 120 # Give up on a single protein after 2 minutes

    if not os.path.exists(soundfont_path):
        print(f"FATAL ERROR: SoundFont file not found at '{soundfont_path}'")
        return

    path_configs = [
        {'input': 'fasta/train', 'wav_out': 'train_temp_wav', 'img_out': 'train_image'}
        #{'input': 'fasta/test',  'wav_out': 'test_temp_wav',  'img_out': 'test_image'}
    ]
  
    num_processes = max(1, os.cpu_count() - 1) 
    all_error_messages = []
    tasks = []

    print(f"Using {num_processes} processes for parallel execution.")
    print(f"Setting a timeout of {TASK_TIMEOUT_SECONDS} seconds per protein.")
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
            
            specific_temp_dir = os.path.join(base_wav_dir, file_base_name)
            specific_img_dir = os.path.join(base_img_dir, file_base_name)
            os.makedirs(specific_temp_dir, exist_ok=True)
            os.makedirs(specific_img_dir, exist_ok=True)

            try:
                valid_sequences_in_file = 0
                with open(input_fasta_path, "r") as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        if valid_sequences_in_file >= processing_limit:
                            print(f"  - Reached limit of {processing_limit} for file '{filename}'.")
                            break

                        if 'X' in str(record.seq).upper():
                            continue

                        sanitized_id = record.id.replace('|', '_').replace(':', '_').replace('/', '_').replace('\\', '_')
                        task_args = (record, sanitized_id, specific_temp_dir, specific_img_dir, soundfont_path)
                        tasks.append(task_args)
                        valid_sequences_in_file += 1
                
                if valid_sequences_in_file == 0:
                     print(f"  - No valid sequences found in '{filename}'.")

            except Exception as e:
                all_error_messages.append(f"[ERROR] Could not parse file '{filename}'. It may be corrupted. Skipping. Details: {e}")

    if not tasks:
        print("\nNo valid protein sequences found to process across all files.")
    else:
        print(f"\n--- Collected a total of {len(tasks)} valid sequences. Starting processing. ---")
        
        # NEW: Use apply_async and collect results with a timeout
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Submit all tasks and get a list of AsyncResult objects
            results = [pool.apply_async(process_fasta_record, args=(task,)) for task in tasks]
            
            # Use tqdm to iterate over the results and retrieve them
            pbar = tqdm(total=len(results), desc="Processing sequences")
            for i, res in enumerate(results):
                try:
                    # Wait for the result, but with a timeout
                    error = res.get(timeout=TASK_TIMEOUT_SECONDS)
                    if error:
                        all_error_messages.append(error)
                except multiprocessing.TimeoutError:
                    # This is the key part: handle the timeout
                    protein_name = tasks[i][1] # Get the protein name from the original task list
                    all_error_messages.append(f"[TIMEOUT] Protein '{protein_name}' took too long to process and was skipped.")
                except Exception as e:
                    # Handle other potential errors during result retrieval
                    protein_name = tasks[i][1]
                    all_error_messages.append(f"[ERROR] An unexpected error occurred while getting result for protein '{protein_name}': {e}")
                finally:
                    # Update the progress bar regardless of outcome
                    pbar.update(1)
            pbar.close()
          
    print("\n--- All processing complete. ---")
  
    if all_error_messages:
        print("\n\n========================================")
        print("         SUMMARY OF ERRORS/SKIPS")
        print("========================================")
        for i, msg in enumerate(sorted(all_error_messages)):
            print(f"\n{i+1}. {msg}")
        print("\n========================================")
    else:
        print("\nProcessing finished with no errors reported.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()