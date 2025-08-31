import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import multiprocessing
from tqdm import tqdm
import traceback
import contextlib
import gc
import re
from midiutil import MIDIFile
from midi2audio import FluidSynth

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
    pitches = [map_rd_scale.get(aa, 'Rest') for aa in chain_seq]
    timbres = [map_rd_ium.get(aa, 0) for aa in chain_seq]
    rhythms = [map_rd_rhythm.get(aa, 480) for aa in chain_seq]
    return pitches, timbres, rhythms

def save_music_data_to_midi(music_data, output_path, protein_name):
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
        return False, f"MIDI Write Error: {e}"

def convert_midi_to_wav(midi_file_path, wav_file_path, soundfont_path, sample_rate=16000):
    try:
        fs = FluidSynth(sound_font=soundfont_path, sample_rate=sample_rate)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                fs.midi_to_audio(midi_file_path, wav_file_path)
        return True, None
    except Exception as e:
        return False, f"FluidSynth Error: {e}"

def generate_spectrogram(wav_file_path, output_image_path):
    try:
        y, sr = librosa.load(wav_file_path, sr=None)
        if np.max(np.abs(y)) == 0:
            return False, "WAV file is silent"
        
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
        return False, f"Spectrogram Error: {e}"

def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        header, current_seq = None, ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and current_seq:
                    sequences.append((header, current_seq))
                header = line[1:].split()[0]
                current_seq = ""
            else:
                current_seq += line
        if header and current_seq:
            sequences.append((header, current_seq))
    return sequences

def process_sequence(args):
    header, chain_seq, spectrogram_output_dir, soundfont_path, temp_dir = args
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "_", header)
    midi_file_path = os.path.join(temp_dir, f"{sanitized_name}.mid")
    wav_file_path = os.path.join(temp_dir, f"{sanitized_name}.wav")
    spectrogram_path = os.path.join(spectrogram_output_dir, f"{sanitized_name}.png")
    try:
        chain_seq = chain_seq.upper().replace('*', '')
        if not chain_seq or 'X' in chain_seq or 'B' in chain_seq or 'Z' in chain_seq:
            return f"[SKIPPED] Protein '{header}': Invalid characters or empty."
        music_data = convert_to_music(chain_seq)
        if all(p == 'Rest' for p in music_data[0]):
            return f"[SKIPPED] Protein '{header}': No mappable amino acids."
        success, err = save_music_data_to_midi(music_data, midi_file_path, sanitized_name)
        if not success: return f"[FAILED] Protein '{header}' at MIDI: {err}"
        success, err = convert_midi_to_wav(midi_file_path, wav_file_path, soundfont_path)
        if not success: return f"[FAILED] Protein '{header}' at WAV: {err}"
        if os.path.exists(wav_file_path):
            success, err = generate_spectrogram(wav_file_path, spectrogram_path)
            if not success: return f"[FAILED] Protein '{header}' at Spectrogram: {err}"
        else:
            return f"[FAILED] Protein '{header}': WAV file not created."
        return None
    except Exception as e:
        return f"[UNEXPECTED ERROR] Protein '{header}': {e}\n{traceback.format_exc()}"
    finally:
        for f_path in [midi_file_path, wav_file_path]:
            if os.path.exists(f_path):
                try: os.remove(f_path)
                except OSError: pass
        gc.collect()

# --- MAIN EXECUTION BLOCK (WITH DIAGNOSTICS) ---
def main():
    soundfont_path = "GeneralUser.sf2"
    batch_size = 200
    TASK_TIMEOUT_SECONDS = 60
    num_processes = os.cpu_count() or 4
    jobs = [
        {'name': 'Train', 'input_dir': 'fasta/train', 'output_dir': 'train_image', 'target_count': 800},
        {'name': 'Test',  'input_dir': 'fasta/test',  'output_dir': 'test_image',  'target_count': 200}
    ]
    if not os.path.exists(soundfont_path):
        print(f"FATAL ERROR: SoundFont file not found at '{soundfont_path}'")
        return
    print(f"Using {num_processes} processes for parallel execution.")
    all_error_messages = []
    for job in jobs:
        print("\n" + "="*80)
        print(f"--- Starting processing for: {job['name']} Set ---")
        print(f"Input: '{job['input_dir']}', Output: '{job['output_dir']}', Target per class: {job['target_count']}")
        print("="*80)
        if not os.path.isdir(job['input_dir']):
            print(f"[WARNING] Input directory '{job['input_dir']}' not found. Skipping this set.")
            continue
        fasta_files = sorted([f for f in os.listdir(job['input_dir']) if f.endswith('.fasta')])
        for fasta_filename in fasta_files:
            class_label = os.path.splitext(fasta_filename)[0]
            input_fasta_path = os.path.join(job['input_dir'], fasta_filename)
            output_image_dir = os.path.join(job['output_dir'], class_label)
            temp_files_dir = os.path.join(output_image_dir, 'temp')
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(temp_files_dir, exist_ok=True)
            print(f"\n--- Processing Class: {class_label} ({job['name']}) ---")
            try:
                sequences_to_process = parse_fasta(input_fasta_path)
                
                # =============================================================
                # +++ NEW DIAGNOSTIC LINE +++
                # 这会告诉你脚本在每个文件中找到了多少序列
                print(f">>> Diagnostic: Found {len(sequences_to_process)} sequences in '{input_fasta_path}'.")
                # =============================================================

            except Exception as e:
                msg = f"[ERROR] Could not read or parse FASTA file '{input_fasta_path}'. Details: {e}"
                print(msg)
                all_error_messages.append(msg)
                continue

            if not sequences_to_process:
                print(">>> No valid sequences found to process. Skipping to next class.")
                continue

            tasks = [ (header, seq, output_image_dir, soundfont_path, temp_files_dir) for header, seq in sequences_to_process]
            successful_count = 0
            for i in range(0, len(tasks), batch_size):
                if successful_count >= job['target_count']:
                    print(f"\nTarget of {job['target_count']} reached for Class {class_label}. Moving to next class.")
                    break
                batch_tasks = tasks[i:i + batch_size]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    desc = f"Class {class_label} (Success: {successful_count}/{job['target_count']})"
                    pbar = tqdm(total=len(batch_tasks), desc=desc, leave=False)
                    results = [pool.apply_async(process_sequence, args=(task,)) for task in batch_tasks]
                    for future in results:
                        if successful_count >= job['target_count']: break
                        try:
                            error = future.get(timeout=TASK_TIMEOUT_SECONDS)
                            if error: all_error_messages.append(error)
                            else: successful_count += 1
                        except multiprocessing.TimeoutError:
                            all_error_messages.append(f"[TIMEOUT] A task exceeded the {TASK_TIMEOUT_SECONDS}s limit.")
                        except Exception as e:
                            all_error_messages.append(f"[FUTURE ERROR] An unexpected error occurred: {e}")
                        pbar.set_description(f"Class {class_label} (Success: {successful_count}/{job['target_count']})")
                        pbar.update(1)
                    pbar.close()
            try:
                if os.path.exists(temp_files_dir): os.rmdir(temp_files_dir)
            except OSError: pass
    print("\n\n--- ALL PROCESSING COMPLETE ---")
    if all_error_messages:
        print(f"\nEncountered {len(all_error_messages)} errors/skips during the process.")
        error_log_path = "conversion_error_log.txt"
        with open(error_log_path, 'w') as f:
            for msg in all_error_messages: f.write(f"{msg}\n")
        print(f"Full details saved to: {error_log_path}")
    else:
        print("\nProcessing finished with no errors reported.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()