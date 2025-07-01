import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.colors as colors
from Bio import SeqIO
from midiutil import MIDIFile
from midi2audio import FluidSynth

# 定义映射关系，将氨基酸序列映射为音高、音色和节奏
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

# 将氨基酸链转换为音乐数据
def convert_to_music(chain_seq):
    pitches = [map_rd_scale.get(aa, 'Rest') for aa in chain_seq]  # 获取音高
    timbres = [map_rd_ium.get(aa, 0) for aa in chain_seq]  # 获取音色
    rhythms = [map_rd_rhythm.get(aa, 480) for aa in chain_seq]  # 获取节奏
    return pitches, timbres, rhythms

# 保存音乐数据为MIDI文件
def save_music_data_to_midi(music_data, output_directory, output_name):
    file_path = os.path.join(output_directory, f"{output_name}.mid")  # 生成MIDI文件路径
    
    midi = MIDIFile(1)
    track = 0
    time = 0
    midi.addTrackName(track, time, f"{output_name} Music")  # 添加曲目名称
    midi.addTempo(track, time, 120)  # 设置节拍速度
    
    pitches, timbres, rhythms = music_data
    for pitch, timbre, rhythm in zip(pitches, timbres, rhythms):
        if pitch != 'Rest':
            note = int(pitch[1]) * 12 + (ord(pitch[0]) - ord('C'))
            duration = rhythm / 480
            midi.addNote(track, timbre, note, time, duration, 100)  # 添加音符到MIDI文件
            time += duration
    
    with open(file_path, 'wb') as output_file:
        midi.writeFile(output_file)  # 保存MIDI文件
    print(f"Music data saved to {file_path}")
    return file_path

# 将MIDI文件转换为WAV文件
def convert_midi_to_wav(midi_file_path, wav_file_path, soundfont_path, sample_rate=44100):
    try:
        fs = FluidSynth(soundfont_path, sample_rate=sample_rate)
        fs.midi_to_audio(midi_file_path, wav_file_path)  # 转换为WAV文件
        print(f"Conversion to WAV completed: {wav_file_path}")
    except Exception as e:
        print(f"Error converting {midi_file_path} to WAV: {e}")

# 生成并保存频谱图
def generate_spectrogram(wav_file_path, output_directory, output_name):
    y, sr = librosa.load(wav_file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 创建自定义颜色映射
    white = np.array([1, 1, 1, 1])  # 白色
    blue = np.array([0, 0, 1, 1])  # 蓝色
    n_bins = 256
    color_array = np.array([white + (blue - white) * (i/n_bins)**0.5 for i in range(n_bins)])
    custom_cmap = colors.ListedColormap(color_array)
    custom_cmap.set_under('white')
    
    plt.figure(figsize=(12, 6))
    plt.axis('off')  # 关闭坐标轴显示
    librosa.display.specshow(S_dB, sr=sr, cmap=custom_cmap, vmin=-80, vmax=0)
    
    output_file_path = os.path.join(output_directory, f"{output_name}.png")  # 保存频谱图路径
    plt.savefig(output_file_path, facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f'Spectrogram saved to {output_file_path}')

# 批量处理目录中的所有FASTA文件
def process_fasta_files_in_directory(input_directory, soundfont_path):
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.fasta') or file_name.endswith('.fa'):
            file_path = os.path.join(input_directory, file_name)
            base_name = os.path.splitext(file_name)[0]  # 获取文件的基本名称（不含扩展名）

            # 创建输出目录
            midi_output_directory = os.path.join(input_directory, 'wav', base_name)
            spectrogram_output_directory = os.path.join(input_directory, 'image', base_name)
            os.makedirs(midi_output_directory, exist_ok=True)
            os.makedirs(spectrogram_output_directory, exist_ok=True)

            process_fasta_to_wav_and_spectrogram(
                file_path, midi_output_directory, spectrogram_output_directory, base_name, soundfont_path)

# 处理单个FASTA文件，生成WAV和频谱图
def process_fasta_to_wav_and_spectrogram(input_fasta, midi_output_directory, spectrogram_output_directory, base_name, soundfont_path, max_proteins=220):
    protein_count = 0
    
    for record in SeqIO.parse(input_fasta, "fasta"):
        if protein_count >= max_proteins:
            print(f"Reached maximum of {max_proteins} proteins for {input_fasta}. Moving to next file.")
            break
        
        # 命名规则：原文件名 + (序号)
        protein_name = f"{base_name}({protein_count + 1})"
        chain_seq = str(record.seq)
        music_data = convert_to_music(chain_seq)
        
        # 保存MIDI文件
        midi_file_path = save_music_data_to_midi(music_data, midi_output_directory, protein_name)
        
        # 转换MIDI文件为WAV文件
        wav_file_path = os.path.join(midi_output_directory, f"{protein_name}.wav")
        convert_midi_to_wav(midi_file_path, wav_file_path, soundfont_path, sample_rate=16000)
        
        # 生成并保存频谱图
        generate_spectrogram(wav_file_path, spectrogram_output_directory, protein_name)
        
        protein_count += 1

# 主函数，调用批量处理
def main():
    input_directory = "/mnt/musicnn-master/more"  # 输入FASTA文件所在的文件夹路径
    soundfont_path = "/mnt/musicnn-master/GeneralUser GS 1.471/GeneralUser GS v1.471.sf2"  # SoundFont文件路径
    
    process_fasta_files_in_directory(input_directory, soundfont_path)

if __name__ == "__main__":
    main()
