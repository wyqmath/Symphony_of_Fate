import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from scipy.stats import mode
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from tqdm import tqdm

# --- Matplotlib Configuration (使用更鲜明的颜色) ---
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.unicode_minus'] = False

# --- Global Constants and Hyperparameters ---
CLASS_NAMES = [
    'Enzyme', 'Structural', 'Transport', 'Storage',
    'Signalling', 'Receptor', 'Gene Regulatory',
    'Immune', 'Chaperone'
]
NUM_CLASSES = len(CLASS_NAMES)

# Data Directories
fasta_train_dir = os.path.join('fasta', 'train')
fasta_test_dir = os.path.join('fasta', 'test')

# BLAST + k-NN Hyperparameters
K_NEIGHBORS = 5  # k-NN中的k值
EXPERIMENT_NAME = f"BLAST_kNN_k{K_NEIGHBORS}"

# ==============================================================================
# SECTION 2: FASTA DATA LOADING
# ==============================================================================
def load_fasta_data(fasta_dir, limit_per_file=None):
    """
    加载FASTA文件，返回序列记录列表和标签字典。
    修改为返回SeqRecord对象列表，以保留序列ID。
    """
    sequences = []
    labels = []
    print(f"--- Loading FASTA data from {fasta_dir} ---")
    if not os.path.exists(fasta_dir):
        print(f"Warning: Directory not found: {fasta_dir}")
        return sequences, labels

    for filename in tqdm(os.listdir(fasta_dir), desc=f"Reading files from {os.path.basename(fasta_dir)}"):
        if not filename.endswith(('.fasta', '.fa')):
            continue
        try:
            label = int(os.path.splitext(filename)[0]) - 1
        except ValueError:
            continue

        filepath = os.path.join(fasta_dir, filename)
        sequences_in_file = 0
        for record in SeqIO.parse(filepath, "fasta"):
            if limit_per_file and sequences_in_file >= limit_per_file:
                break
            # 过滤掉包含未知氨基酸'X'的序列
            if 'X' in record.seq:
                continue
            sequences.append(record)
            labels.append(label)
            sequences_in_file += 1
            
    return sequences, np.array(labels)

# ==============================================================================
# SECTION 3: BLAST DATABASE AND PREDICTION FUNCTIONS
# ==============================================================================

def create_blast_db(train_sequences, db_path):
    """
    使用训练序列创建一个BLAST蛋白质数据库。
    """
    print("\n--- Creating BLAST Database from training sequences ---")
    # 1. 将训练集序列写入一个临时的FASTA文件
    train_fasta_path = f"{db_path}.fasta"
    with open(train_fasta_path, "w") as f_out:
        SeqIO.write(train_sequences, f_out, "fasta")

    # 2. 调用 makeblastdb 命令
    # -dbtype prot: 指定数据库类型为蛋白质
    # -in: 输入的FASTA文件
    # -out: 输出的数据库路径和前缀
    command = [
        'makeblastdb',
        '-dbtype', 'prot',
        '-in', train_fasta_path,
        '-out', db_path
    ]
    
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"BLAST database created successfully at '{db_path}'")
        return True
    except FileNotFoundError:
        print("\n[ERROR] `makeblastdb` command not found.")
        print("Please ensure NCBI BLAST+ is installed and its 'bin' directory is in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] `makeblastdb` failed with error:")
        print(e.stderr)
        return False

def predict_with_blast_knn(test_sequences, y_test, train_id_to_label, db_path, k):
    """
    使用BLAST+kNN对测试集进行预测。
    """
    print(f"\n--- Starting Prediction using BLAST + k-NN (k={k}) ---")
    y_pred = []
    
    for i, test_record in enumerate(tqdm(test_sequences, desc="Predicting")):
        # 1. 将单个测试序列写入临时查询文件
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".fa") as query_file:
            SeqIO.write(test_record, query_file, "fasta")
            query_filepath = query_file.name

        # 2. 运行 blastp
        # -query: 查询文件
        # -db: 目标数据库
        # -outfmt 6: 表格输出格式，易于解析
        # -max_target_seqs: 限制返回的最佳匹配数量，设为k即可
        # -evalue: E-value阈值
        command = [
            'blastp',
            '-query', query_filepath,
            '-db', db_path,
            '-outfmt', '6 sseqid bitscore',  # 只输出目标序列ID和比特分
            '-max_target_seqs', str(k),
            '-evalue', '1e-5'
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # 3. 解析BLAST结果
            hits = result.stdout.strip().split('\n')
            neighbor_ids = [line.split('\t')[0] for line in hits if line]

            if not neighbor_ids:
                # 如果没有找到任何相似的序列，则随机猜测一个类别
                # 这是一个简化的处理方式，也可以选择最常见的类别等
                predicted_label = np.random.choice(NUM_CLASSES)
            else:
                # 4. 获取邻居的标签并投票
                neighbor_labels = [train_id_to_label[nid] for nid in neighbor_ids]
                # 使用scipy.stats.mode找到最常见的标签
                predicted_label = mode(neighbor_labels, keepdims=False).mode

            y_pred.append(predicted_label)

        except subprocess.CalledProcessError as e:
            # 如果blastp出错，也进行一次随机猜测
            print(f"\nWarning: blastp failed for sequence {test_record.id}. Error: {e.stderr}")
            y_pred.append(np.random.choice(NUM_CLASSES))
        finally:
            # 清理临时查询文件
            os.remove(query_filepath)
            
    return np.array(y_test), np.array(y_pred)

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    # --- 1. Load data ---
    train_seqs, y_train = load_fasta_data(fasta_train_dir, limit_per_file=800)
    test_seqs, y_test = load_fasta_data(fasta_test_dir, limit_per_file=200)

    if not train_seqs or not test_seqs:
        print("Data loading failed. Exiting.")
    else:
        # 创建一个从训练序列ID到其标签的映射字典，用于后续投票
        train_id_to_label_map = {record.id: label for record, label in zip(train_seqs, y_train)}

        # --- 2. Create a temporary directory for BLAST database ---
        # 使用tempfile确保脚本结束后临时文件被清理
        with tempfile.TemporaryDirectory() as temp_dir:
            db_name = os.path.join(temp_dir, "protein_db")
            
            # --- 3. Build BLAST database ---
            db_created = create_blast_db(train_seqs, db_name)
            
            if db_created:
                # --- 4. Run prediction ---
                y_true, y_pred = predict_with_blast_knn(
                    test_seqs, y_test, train_id_to_label_map, db_name, k=K_NEIGHBORS
                )
                
                # --- 5. Report and Visualize Results ---
                total_time = time.time() - start_time
                accuracy = accuracy_score(y_true, y_pred)
                
                print("\n" + "="*30)
                print(f"FINAL RESULTS for {EXPERIMENT_NAME}")
                print("="*30)
                print(f"Total execution time: {total_time:.2f} seconds.")
                print(f'Test Set Accuracy: {accuracy * 100:.2f}%')
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

                # --- Plot and save confusion matrix (with new color) ---
                print('Plotting and saving Confusion Matrix...')
                fig, ax = plt.subplots(figsize=(12, 10))
                cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
                # 修改点：使用了新的颜色映射 'GnBu' (Green-Blue)
                disp.plot(cmap='GnBu', values_format='d', ax=ax, xticks_rotation='vertical')
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                ax.set_title(f'Confusion Matrix ({EXPERIMENT_NAME})', fontsize=16, pad=20)
                fig.tight_layout()
                filename = f"Confusion_Matrix_{EXPERIMENT_NAME}.png"
                fig.savefig(filename, bbox_inches='tight', dpi=300)
                print(f"Saved confusion matrix plot to: {filename}")
                plt.close(fig)