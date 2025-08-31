import os
import cv2
import numpy as np
import librosa
import random
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.svm import SVC
from xgboost import XGBClassifier
import concurrent.futures
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib
from Bio import SeqIO
from collections import Counter
from tqdm import tqdm

# --- Matplotlib Configuration ---
# 使用一个不需要GUI并且可以保存文件的后端
matplotlib.use('Agg')
# 使用一个默认的、可用的字体来防止 'findfont' 错误
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

# --- Class Names ---
CLASS_NAMES = [
    'Enzyme', 'Structural', 'Transport', 'Storage',
    'Signalling', 'Receptor', 'Gene Regulatory',
    'Immune', 'Chaperone'
]

# --- Data Directories ---
# 此脚本假定从一个根目录运行，该根目录包含
# 'train_image', 'test_image', 和 'fasta' 作为子目录。
spectrogram_train_dir = 'train_image'
spectrogram_test_dir = 'test_image'
fasta_train_dir = os.path.join('fasta', 'train')
fasta_test_dir = os.path.join('fasta', 'test')


# ==============================================================================
# SECTION 1: MUSIC ENCODING DATA PROCESSING
# ==============================================================================

def data_augmentation(img):
    """对单个图像进行数据增强。"""
    augmented_images = [img]
    # 添加高斯噪声
    noise = np.random.normal(0.0001, 0.005, img.shape)
    augmented_images.append(np.clip(img + noise, 0, 1))
    # 水平翻转
    augmented_images.append(np.fliplr(img))
    # 调整对比度和亮度
    contrast_adjusted = img * random.uniform(0.8, 1.2)
    brightness_adjusted = contrast_adjusted + random.uniform(-0.05, 0.05)
    augmented_images.append(np.clip(brightness_adjusted, 0, 1))
    return augmented_images

def extract_features_from_spectrogram(spectrogram):
    """从频谱图中提取MFCC特征。"""
    mfccs = librosa.feature.mfcc(S=librosa.db_to_power(spectrogram), sr=22050, n_mfcc=60)
    return np.mean(mfccs, axis=1)

def process_image_for_music_encoding(img_path, label, augment=False):
    """处理单个图像文件以进行音乐编码特征提取。"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
        images_to_process = data_augmentation(img) if augment else [img]
        features = [extract_features_from_spectrogram(proc_img) for proc_img in images_to_process]
        return features, int(label) - 1
    return None

def load_music_encoding_features(folder, augment=False):
    """从包含频谱图图像的文件夹中加载音乐编码特征。"""
    features, labels = [], []
    aug_status = "with augmentation" if augment else "without augmentation"
    print(f"Loading music encoding features from {folder} {aug_status}...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = []
        if not os.path.exists(folder):
            print(f"Warning: Directory not found: {folder}")
            return np.array(features), np.array(labels)
        for label_folder in os.listdir(folder):
            # --- 新增修改：跳过 .ipynb_checkpoints 文件夹 ---
            if label_folder == '.ipynb_checkpoints':
                print(f"Skipping ipynb_checkpoints directory: {os.path.join(folder, label_folder)}")
                continue
            # --- 修改结束 ---
            label_path = os.path.join(folder, label_folder)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    img_path = os.path.join(label_path, filename)
                    tasks.append((img_path, label_folder, augment))
        futures = [executor.submit(process_image_for_music_encoding, *task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing images in {os.path.basename(folder)}"):
            result = future.result()
            if result:
                img_features, img_label = result
                features.extend(img_features)
                labels.extend([img_label] * len(img_features))
    print(f"Loaded {len(features)} samples.")
    return np.array(features), np.array(labels)

# ==============================================================================
# SECTION 2: FASTA DATA PROCESSING
# ==============================================================================

KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'U': 0.0, 'X': 0.0, 'B': -3.5, 'Z': -3.5, 'J': 3.8
}
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'W': 11, 'Y': 2, 'V': -3},
    'Y': {'Y': 7, 'V': -1},
    'V': {'V': 4}
}

def _load_sequences_with_limit(fasta_dir, limit_per_file):
    """从FASTA目录加载序列，每个文件有序列数量限制。"""
    sequences, labels = [], []
    print(f"Reading FASTA files from: {fasta_dir} (limit: {limit_per_file} sequences per file)")
    if not os.path.exists(fasta_dir):
        print(f"Warning: Directory not found: {fasta_dir}")
        return sequences, labels
    
    file_list = os.listdir(fasta_dir)
    for filename in tqdm(file_list, desc=f"Reading files from {os.path.basename(fasta_dir)}"):
        # --- 新增修改：跳过 .ipynb_checkpoints 文件夹 ---
        if filename == '.ipynb_checkpoints':
            print(f"Skipping ipynb_checkpoints directory: {os.path.join(fasta_dir, filename)}")
            continue
        # --- 修改结束 ---
        if not filename.endswith(('.fasta', '.fa')):
            continue
        try:
            label = int(os.path.splitext(filename)[0]) - 1
        except ValueError:
            print(f"Warning: Could not determine label from filename '{filename}'. Skipping.")
            continue
        filepath = os.path.join(fasta_dir, filename)
        sequences_in_file = 0
        for record in SeqIO.parse(filepath, "fasta"):
            if sequences_in_file >= limit_per_file:
                break
            seq = str(record.seq).upper()
            if 'X' in seq:
                continue
            sequences.append(seq)
            labels.append(label)
            sequences_in_file += 1
    print(f"Found {len(sequences)} valid sequences in {len(np.unique(labels))} classes.")
    return sequences, labels

def _process_sequences_to_features(sequences, max_len, apply_noise=False, noise_level=0.05):
    """将序列转换为FFT特征。"""
    features = []
    for seq in tqdm(sequences, desc="Generating FFT features", leave=False):
        hydro_signal = np.array([KYTE_DOOLITTLE.get(aa, 0) for aa in seq])
        if apply_noise:
            noise = np.random.normal(0, noise_level, hydro_signal.shape)
            hydro_signal += noise
        # 截断长于max_len的序列
        if len(hydro_signal) > max_len:
            hydro_signal = hydro_signal[:max_len]
        # 填充短于max_len的序列
        padded_signal = np.pad(hydro_signal, (0, max_len - len(hydro_signal)), 'constant')
        fft_result = np.fft.fft(padded_signal)
        power_spectrum = np.abs(fft_result)**2
        features.append(power_spectrum[:max_len // 2 + 1])
    return np.array(features)

def _conservative_mutation(sequence, num_mutations=2):
    """对序列进行保守突变。"""
    seq_list = list(sequence)
    possible_indices = list(range(len(seq_list)))
    random.shuffle(possible_indices)
    mutated_count = 0
    for i in possible_indices:
        if mutated_count >= num_mutations: break
        original_aa = seq_list[i]
        if original_aa not in BLOSUM62: continue
        substitutions = [target_aa for target_aa, score in BLOSUM62[original_aa].items() if score > 0 and target_aa != original_aa]
        if substitutions:
            seq_list[i] = random.choice(substitutions)
            mutated_count += 1
    return "".join(seq_list)

def _random_cropping(sequence, crop_ratio=0.9):
    """对序列进行随机裁剪。"""
    original_len = len(sequence)
    crop_len = int(original_len * crop_ratio)
    if crop_len >= original_len: return sequence
    start_index = random.randint(0, original_len - crop_len)
    return sequence[start_index : start_index + crop_len]

def load_fasta_features_with_augmentation(train_dir, test_dir):
    """加载并从FASTA文件提取增强特征。"""
    print("\nLoading and extracting ENHANCED features from FASTA files...")
    train_seqs, y_train_orig = _load_sequences_with_limit(train_dir, limit_per_file=800)
    test_seqs, y_test = _load_sequences_with_limit(test_dir, limit_per_file=200)

    if not train_seqs:
        print("Error: No training sequences found. Skipping FASTA experiment.")
        return np.array([]), np.array([]), np.array([]), np.array([])
    if not test_seqs:
        print("Warning: No test sequences found. Proceeding with training data only.")

    print("Augmenting training sequences...")
    aug_train_seqs, aug_train_labels = [], []
    for seq, label in tqdm(zip(train_seqs, y_train_orig), total=len(train_seqs), desc="Augmenting sequences"):
        aug_train_seqs.append(seq)
        aug_train_labels.append(label)
        aug_train_seqs.append(_conservative_mutation(seq))
        aug_train_labels.append(label)
        aug_train_seqs.append(_random_cropping(seq))
        aug_train_labels.append(label)
    print(f"Original training sequences: {len(train_seqs)}. Augmented training sequences: {len(aug_train_seqs)}.")

    max_len = max(len(s) for s in aug_train_seqs) if aug_train_seqs else 0
    if max_len == 0:
        print("Error: max_len is 0, cannot process features. Aborting FASTA experiment.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    print(f"Max protein length from AUGMENTED TRAINING data: {max_len}.")
    print("All train and test sequences will be padded or truncated to this length.")

    X_train = _process_sequences_to_features(aug_train_seqs, max_len, apply_noise=True)
    X_test = _process_sequences_to_features(test_seqs, max_len, apply_noise=False)
    
    return X_train, np.array(aug_train_labels), X_test, np.array(y_test)

# ==============================================================================
# SECTION 3: UNIFIED MACHINE LEARNING PIPELINE (MODIFIED)
# ==============================================================================

def run_ml_pipeline(X_train, y_train, X_test, y_test, experiment_name, class_labels, confusion_matrix_cmap):
    """运行统一的机器学习流程。"""
    print("\n" + "="*80)
    print(f"RUNNING PIPELINE FOR: {experiment_name}")
    print("="*80)

    if X_train.shape[0] == 0:
        print("Training data is empty. Skipping pipeline.")
        return
    if X_test.shape[0] == 0:
        print("Test data is empty. Skipping pipeline.")
        return

    # --- 两步特征选择 ---
    print("\n--- Running Two-Step Feature Selection ---")
    print(f"Initial number of features: {X_train.shape[1]}")

    # 步骤 1: 使用SelectFromModel进行粗略筛选
    print("Step 1: Applying SelectFromModel for broad feature reduction...")
    sfm_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # 修改：降低阈值以保留更多特征
    sfm = SelectFromModel(sfm_estimator, threshold="0.25*median", prefit=False)
    X_train_sfm = sfm.fit_transform(X_train, y_train)
    X_test_sfm = sfm.transform(X_test)
    print(f"Features after SelectFromModel: {X_train_sfm.shape[1]}")

    # 步骤 2: 在筛选后的特征集上使用RFE进行精细选择
    print("Step 2: Applying RFE for fine-grained feature selection...")
    if X_train_sfm.shape[1] <= 1:
        print("Not enough features for RFE. Using features from SelectFromModel.")
        X_train_rfe, X_test_rfe = X_train_sfm, X_test_sfm
    else:
        rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        n_features_to_select = min(50, X_train_sfm.shape[1])
        if n_features_to_select <= 0:
             print("No features to select. Skipping RFE.")
             X_train_rfe, X_test_rfe = X_train_sfm, X_test_sfm
        else:
            rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select, step=0.1)
            X_train_rfe = rfe.fit_transform(X_train_sfm, y_train)
            X_test_rfe = rfe.transform(X_test_sfm)
    print(f"Final features selected after RFE: {X_train_rfe.shape[1]}")
    print(f"Training data shape after feature selection: {X_train_rfe.shape}")
    print(f"Test data shape after feature selection: {X_test_rfe.shape}")

    # --- 定义用于集成的未训练基础模型 (优化参数) ---
    # 修改：移除max_depth并降低min_samples_split以创建更深、更详细的树
    rf_clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
    
    # 修改：增加学习率以加快学习速度，并减少正则化 (lambda/alpha)
    xgb_clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, reg_lambda=0.5, alpha=0, random_state=42, eval_metric='mlogloss', tree_method='hist', device='cuda')
    
    # 修改：显著增加C以减少正则化，允许更复杂的决策边界
    svm_clf = SVC(kernel='rbf', C=10, probability=True, random_state=42)
    
    # 修改：增加神经元数量以增强模型容量
    mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=2000, random_state=42, alpha=0.05, early_stopping=True)

    # --- 创建集成模型 (投票分类器) ---
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_clf), ('xgb', xgb_clf), ('svm', svm_clf), ('mlp', mlp_clf)],
        voting='soft', weights=[1, 1.5, 1, 1], n_jobs=-1 # 稍微增加强大的XGBoost模型的权重
    )

    # --- 执行交叉验证 ---
    print("\n--- Evaluating ENSEMBLE model with Cross-Validation ---")
    skf_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 此处使用 n_jobs=1 以避免嵌套并行问题
    cv_scores = cross_val_score(voting_clf, X_train_rfe, y_train, cv=skf_eval, scoring='accuracy', n_jobs=1)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"Ensemble Cross-Validation Accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    print("(This is a realistic estimate of the model's performance on unseen data)")

    # --- 训练最终的集成模型 ---
    print('\n--- Training final ENSEMBLE model on the full training set... ---')
    print(f"Class distribution for final training: {Counter(y_train)}")
    start_time = time.time()
    voting_clf.fit(X_train_rfe, y_train)
    train_time = time.time() - start_time
    print(f"Final ensemble training completed in {train_time:.2f} seconds.")
    
    # --- 在测试集上评估最终模型 ---
    print('\n--- Evaluating final ENSEMBLE model on the test set... ---')
    y_pred = voting_clf.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f"FINAL ENSEMBLE RESULTS for {experiment_name}")
    print("="*30)
    print(f'Test Set Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report:")
    report_labels = np.unique(np.concatenate((y_test, y_pred)))
    report_target_names = [class_labels[i] for i in report_labels]
    print(classification_report(y_test, y_pred, labels=report_labels, target_names=report_target_names, zero_division=0))

    # --- 绘制并保存混淆矩阵 ---
    print('Plotting and saving Confusion Matrix for ENSEMBLE model...')
    cm_labels = np.unique(np.concatenate((y_test, y_pred)))
    cm_display_labels = [class_labels[i] for i in cm_labels]
    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_display_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap=confusion_matrix_cmap, values_format='d', ax=ax, xticks_rotation='vertical')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f'Confusion Matrix ({experiment_name})', fontsize=16, pad=20)
    fig.tight_layout()
    
    filename = f"Confusion_Matrix_{experiment_name.replace(' ', '_')}_Optimized.png"
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved confusion matrix plot to: {filename}")
    plt.close(fig)

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- 实验 1: 音乐编码 ---
    X_train_music, y_train_music = load_music_encoding_features(spectrogram_train_dir, augment=True)
    X_test_music, y_test_music = load_music_encoding_features(spectrogram_test_dir, augment=False)
    
    if X_train_music.size > 0 and y_train_music.size > 0:
        run_ml_pipeline(X_train_music, y_train_music, X_test_music, y_test_music, 
                        "MFCC_ML", 
                        CLASS_NAMES,
                        confusion_matrix_cmap='Blues')
    else:
        print("Could not load data for the Music Encoding experiment. Skipping.")

    # --- 实验 2: FASTA 方法 ---
    X_train_aug, y_train_aug, X_test_aug, y_test_aug = load_fasta_features_with_augmentation(fasta_train_dir, fasta_test_dir)

    if X_train_aug.size > 0 and y_train_aug.size > 0:
        run_ml_pipeline(X_train_aug, y_train_aug, X_test_aug, y_test_aug, 
                        "FASTA_ML", 
                        CLASS_NAMES,
                        confusion_matrix_cmap='Purples')
    else:
        print("Could not load data for the FASTA experiment. Skipping.")