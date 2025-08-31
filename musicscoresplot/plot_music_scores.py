# generate_scores.py (Updated Version)
import os
from music21 import stream, note, duration, environment

# ==============================================================================
# 1. 核心映射规则 (无变化)
# ==============================================================================
map_rd_scale = {
    'A': 'C4', 'R': 'D4', 'N': 'E4', 'D': 'F4', 'C': 'G4', 'E': 'A4', 'Q': 'B4',
    'G': 'C5', 'H': 'D5', 'I': 'E5', 'L': 'F5', 'K': 'G5', 'M': 'A5', 'F': 'B5',
    'P': 'C6', 'S': 'D6', 'T': 'E6', 'W': 'F6', 'Y': 'G6', 'V': 'A6'
}
map_rd_rhythm = {
    'A': 72, 'R': 144, 'N': 216, 'D': 288, 'C': 360, 'E': 432, 'Q': 504,
    'G': 576, 'H': 648, 'I': 720, 'L': 792, 'K': 864, 'M': 936, 'F': 1008,
    'P': 1080, 'S': 1152, 'T': 1224, 'W': 1296, 'Y': 1368, 'V': 1440
}

# ==============================================================================
# 2. 核心转换函数 (修改点 1)
# ==============================================================================

def convert_protein_to_music_data(protein_sequence):
    """将蛋白质序列转换为内部音乐数据结构 (音高列表和节奏列表)"""
    pitches = [map_rd_scale.get(aa, 'Rest') for aa in protein_sequence]
    rhythms = [map_rd_rhythm.get(aa, 480) for aa in protein_sequence]
    return pitches, rhythms

def generate_sheet_music(pitches, rhythms, output_image_path):
    """使用music21从音乐数据生成乐谱图片"""
    try:
        score = stream.Score()
        part = stream.Part()
        for p_name, r_val in zip(pitches, rhythms):
            d = duration.Duration(r_val / 480.0)
            if p_name == 'Rest':
                new_element = note.Rest()
            else:
                new_element = note.Note(p_name)
            new_element.duration = d
            part.append(new_element)
        score.append(part)
        
        print(f"  -> 正在将乐谱渲染为高分辨率图片 (600 DPI): {output_image_path} ...")
        
        # --- MODIFICATION 1: Set DPI for high-quality output ---
        # The .write() method accepts a 'dpi' argument for PNG export.
        score.write('musicxml.png', fp=output_image_path, dpi=600)
        
        print(f"  -> 乐谱已成功保存！")
        return True
        
    except Exception as e:
        print(f"[错误] 无法生成乐谱: {e}")
        print("请确保您已正确安装MuseScore 3并完成了music21的配置。")
        return False

# ==============================================================================
# 3. 主程序 (修改点 2 和 3)
# ==============================================================================

def main():
    """主执行函数"""
    print("--- 蛋白质乐谱生成器 (高分辨率版) ---")
    
    proteins = [
        {
            "id": "1VII",
            "name": "Villin_Headpiece_stable",
            "sequence": "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
        },
        {
            "id": "2LZM",
            "name": "T4_Lysozyme_complex",
            "sequence": "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
        }
    ]

    output_dir = "protein_scores"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: ./{output_dir}/\n")

    for protein in proteins:
        print(f"正在处理蛋白质: {protein['name']} ({protein['id']})")
        
        # --- MODIFICATION 2: Use only the first 30 amino acids ---
        # We slice the sequence to keep the sheet music short and readable for the paper.
        sequence_fragment = protein['sequence'][:30]
        print(f"  -> 使用前30个氨基酸: '{sequence_fragment}'")
        
        # 1. 转换序列片段为音乐数据
        pitches, rhythms = convert_protein_to_music_data(sequence_fragment)
        
        # --- MODIFICATION 3: Update the output filename ---
        # Add a suffix to indicate it's from a 30aa fragment.
        file_path = os.path.join(output_dir, f"{protein['name']}_30aa_600dpi.png")
        
        # 2. 生成乐谱
        generate_sheet_music(pitches, rhythms, file_path)
        print("-" * 20)

if __name__ == "__main__":
    main()