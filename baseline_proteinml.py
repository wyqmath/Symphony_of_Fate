import os
import cv2
import numpy as np
import librosa
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from xgboost import XGBClassifier
import concurrent.futures
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib
from Bio import SeqIO
from collections import Counter

# --- Matplotlib Configuration ---
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

# --- Class Names ---
# User-defined class names for confusion matrix labels
CLASS_NAMES = [
    'Enzyme',
    'Structural Proteins',
    'Transport Proteins',
    'Storage Proteins',
    'Signalling Proteins',
    'Receptor Proteins',
    'Gene Regulatory Proteins',
    'Immune Proteins',
    'Motor Proteins'
]

# --- Data Directories ---
# Please update these paths to your local directories
spectrogram_train_dir = r'C:\Users\Administrator\Desktop\spectrogram\train'
spectrogram_test_dir = r'C:\Users\Administrator\Desktop\spectrogram\test'
fasta_train_dir = r'C:\Users\Administrator\Desktop\spectrogram\train_fasta'
fasta_test_dir = r'C:\Users\Administrator\Desktop\spectrogram\test_fasta'


# ==============================================================================
# SECTION 1: PROPOSED METHOD (MUSIC ENCODING VIA SPECTROGRAMS)
# ==============================================================================

def data_augmentation(img):
    """
    Performs basic data augmentation on a spectrogram image.
    MODIFIED: Removed np.flipud (physically incorrect for spectrograms) and
              added random brightness/contrast adjustment.
    """
    augmented_images = [img]
    # Add random noise
    noise = np.random.normal(0.0001, 0.005, img.shape)
    augmented_images.append(np.clip(img + noise, 0, 1))
    # Flip horizontally (time axis)
    augmented_images.append(np.fliplr(img))
    # Adjust brightness and contrast
    contrast_adjusted = img * random.uniform(0.8, 1.2)
    brightness_adjusted = contrast_adjusted + random.uniform(-0.05, 0.05)
    augmented_images.append(np.clip(brightness_adjusted, 0, 1))
    
    return augmented_images

def extract_features_from_spectrogram(spectrogram):
    """
    Extracts MFCC features from a spectrogram.
    MODIFIED: Reduced n_mfcc from 80 to 40 to lower feature dimensionality
              and reduce the risk of overfitting.
    """
    mfccs = librosa.feature.mfcc(S=librosa.db_to_power(spectrogram), sr=22050, n_mfcc=40)
    return np.mean(mfccs, axis=1)

def process_image_for_music_encoding(img_path, label, augment=False):
    """
    Loads, optionally augments, and extracts features from a single spectrogram image.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
        
        if augment:
            images_to_process = data_augmentation(img)
        else:
            images_to_process = [img]
            
        features = [extract_features_from_spectrogram(proc_img) for proc_img in images_to_process]
        return features, int(label)
    return None

def load_music_encoding_features(folder, augment=False):
    """
    Loads all data for the music encoding method using parallel processing.
    """
    features, labels = [], []
    aug_status = "with augmentation" if augment else "without augmentation"
    print(f"Loading music encoding features from {folder} {aug_status}...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for label in os.listdir(folder):
            label_path = os.path.join(folder, label)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    img_path = os.path.join(label_path, filename)
                    futures.append(executor.submit(process_image_for_music_encoding, img_path, label, augment))
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                img_features, img_label = result
                features.extend(img_features)
                labels.extend([img_label] * len(img_features))
    
    print(f"Loaded {len(features)} samples.")
    return np.array(features), np.array(labels)

# ==============================================================================
# SECTION 2: ENHANCED FASTA PROCESSING METHOD
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

def _load_sequences_from_dir(fasta_dir):
    """Helper function to load all sequences and labels from a directory of FASTA files."""
    sequences = []
    labels = []
    print(f"Reading FASTA files from: {fasta_dir}")
    if not os.path.exists(fasta_dir):
        print(f"Warning: Directory not found: {fasta_dir}")
        return sequences, labels
        
    for filename in os.listdir(fasta_dir):
        if filename.endswith('.fasta') or filename.endswith('.fa'):
            try:
                label = int(os.path.splitext(filename)[0]) - 1
            except ValueError:
                print(f"Warning: Could not determine label from filename '{filename}'. Skipping.")
                continue

            filepath = os.path.join(fasta_dir, filename)
            for record in SeqIO.parse(filepath, "fasta"):
                sequences.append(str(record.seq).upper())
                labels.append(label)
    print(f"Found {len(sequences)} sequences in {len(np.unique(labels))} classes.")
    return sequences, labels

def _process_sequences_to_features(sequences, max_len, apply_noise=False, noise_level=0.05):
    """Helper function to convert a list of sequences into FFT-based feature vectors."""
    features = []
    for seq in sequences:
        hydro_signal = np.array([KYTE_DOOLITTLE.get(aa, 0) for aa in seq])
        
        if apply_noise:
            noise = np.random.normal(0, noise_level, hydro_signal.shape)
            hydro_signal += noise
            
        padded_signal = np.pad(hydro_signal, (0, max_len - len(hydro_signal)), 'constant')
        fft_result = np.fft.fft(padded_signal)
        power_spectrum = np.abs(fft_result)**2
        features.append(power_spectrum[:max_len // 2 + 1])
        
    return np.array(features)

def _conservative_mutation(sequence, num_mutations=2):
    """Augmentation Method 1: Intelligently substitute amino acids."""
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
    """Augmentation Method 2: Extract a continuous segment."""
    original_len = len(sequence)
    crop_len = int(original_len * crop_ratio)
    if crop_len >= original_len: return sequence
    start_index = random.randint(0, original_len - crop_len)
    return sequence[start_index : start_index + crop_len]

def load_fasta_features_with_augmentation(train_dir, test_dir):
    """Main function to load FASTA files and apply augmentation to the training set."""
    print("\nLoading and extracting ENHANCED features from FASTA files...")
    train_seqs, y_train_orig = _load_sequences_from_dir(train_dir)
    test_seqs, y_test = _load_sequences_from_dir(test_dir)

    if not train_seqs or not test_seqs:
        print("Error: No sequences found. Skipping FASTA experiment.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    print("Augmenting training sequences...")
    aug_train_seqs = []
    aug_train_labels = []
    for seq, label in zip(train_seqs, y_train_orig):
        aug_train_seqs.append(seq)
        aug_train_labels.append(label)
        aug_train_seqs.append(_conservative_mutation(seq))
        aug_train_labels.append(label)
        aug_train_seqs.append(_random_cropping(seq))
        aug_train_labels.append(label)
    
    print(f"Original training sequences: {len(train_seqs)}. Augmented training sequences: {len(aug_train_seqs)}.")

    max_len = max(len(s) for s in aug_train_seqs)
    print(f"Max protein length from augmented training set: {max_len}. Padding all signals to this length.")

    X_train = _process_sequences_to_features(aug_train_seqs, max_len, apply_noise=True)
    X_test = _process_sequences_to_features(test_seqs, max_len, apply_noise=False)
    
    return X_train, np.array(aug_train_labels), X_test, np.array(y_test)


# ==============================================================================
# SECTION 3: UNIFIED MACHINE LEARNING PIPELINE
# ==============================================================================

def plot_learning_curve(estimator, X, y, cv, scoring, title):
    """Plots a learning curve for a given estimator."""
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="#4682B4")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#20B2AA")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#4682B4", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#20B2AA", label="Validation Score")
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.ylim(0.5, 1.05)
    plt.show()

def run_ml_pipeline(X_train, y_train, X_test, y_test, experiment_name, class_labels):
    """
    Runs the complete ML pipeline: RFE, SMOTE, Training, and Evaluation.
    MODIFIED: Adjusted RFE and model hyperparameters for better regularization.
    """
    print("\n" + "="*80)
    print(f"RUNNING PIPELINE FOR: {experiment_name}")
    print("="*80)

    if X_train.shape[0] == 0:
        print("Training data is empty. Skipping pipeline.")
        return

    print("Applying Recursive Feature Elimination (RFE)...")
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    # MODIFIED: Select a smaller, fixed number of features to reduce complexity.
    n_features_to_select = min(30, X_train.shape[1])
    if n_features_to_select <= 0:
        print("No features to select. Skipping RFE.")
        X_train_rfe, X_test_rfe = X_train, X_test
    else:
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=0.1)
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
    print(f"Features selected: {X_train_rfe.shape[1]}")

    print(f"Before SMOTE: {X_train_rfe.shape[0]} training samples.")
    min_class_size = min(Counter(y_train).values())
    k_neighbors_for_smote = min(4, min_class_size - 1)

    if k_neighbors_for_smote < 1:
        print(f"Smallest class has size {min_class_size}, which is too small for SMOTE. Skipping SMOTE.")
        X_resampled, y_resampled = X_train_rfe, y_train
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors_for_smote)
        X_resampled, y_resampled = smote.fit_resample(X_train_rfe, y_train)
    print(f"After SMOTE: {X_resampled.shape[0]} training samples.")
    
    # MODIFIED: Increased regularization on RF and MLP to combat overfitting.
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=15, min_samples_leaf=8, max_features='sqrt', random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, reg_lambda=1.5, alpha=0.5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss')
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=2000, random_state=42, alpha=0.5, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)

    print('Creating and training the ensemble model...')
    voting_clf = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('svm', svm), ('mlp', mlp)], voting='soft', weights=[1, 1, 1, 1], n_jobs=-1)
    
    min_samples_per_class_cv = min(Counter(y_resampled).values())
    n_splits = min(5, min_samples_per_class_cv)
    
    if n_splits < 2:
        print(f"Cannot perform cross-validation because the smallest class has only {min_samples_per_class_cv} sample(s). Skipping CV and learning curve.")
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print("Running cross-validation... this may take a while.")
        cv_scores = cross_val_score(voting_clf, X_resampled, y_resampled, cv=skf, scoring='accuracy')
        print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        plot_learning_curve(voting_clf, X_resampled, y_resampled, skf, 'accuracy', f'Learning Curve ({experiment_name})')

    print('Training final model on the full resampled training set...')
    voting_clf.fit(X_resampled, y_resampled)
    
    print('Evaluating on the test set...')
    y_pred = voting_clf.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- FINAL RESULTS ---")
    print(f'Test Set Accuracy for {experiment_name}: {accuracy * 100:.2f}%')
    print("\nClassification Report:")
    report_labels = np.arange(len(class_labels))
    report_target_names = class_labels
    print(classification_report(y_test, y_pred, labels=report_labels, target_names=report_target_names, zero_division=0))

    print('Plotting Confusion Matrix...')
    cm_labels = np.arange(len(class_labels))
    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap='Blues', values_format='d', ax=ax, xticks_rotation='vertical')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f'Confusion Matrix ({experiment_name})', fontsize=16, pad=20)
    fig.tight_layout()
    plt.show()


# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- EXPERIMENT 1: Proposed Method (Music Encoding) ---
    # Apply augmentation ONLY to the training set.
    X_train_music, y_train_music = load_music_encoding_features(spectrogram_train_dir, augment=True)
    X_test_music, y_test_music = load_music_encoding_features(spectrogram_test_dir, augment=False)
    
    if X_train_music.size > 0 and y_train_music.size > 0:
        run_ml_pipeline(X_train_music, y_train_music, X_test_music, y_test_music, 
                        "Proposed Method (Music Encoding)", CLASS_NAMES)
    else:
        print("Could not load data for the Music Encoding experiment. Skipping.")

    # --- EXPERIMENT 2: Enhanced FASTA Method ---
    X_train_aug, y_train_aug, X_test_aug, y_test_aug = load_fasta_features_with_augmentation(fasta_train_dir, fasta_test_dir)

    if X_train_aug.size > 0 and y_train_aug.size > 0:
        run_ml_pipeline(X_train_aug, y_train_aug, X_test_aug, y_test_aug, 
                        "Enhanced FASTA Method", CLASS_NAMES)
    else:
        print("Could not load data for the Enhanced FASTA experiment. Skipping.")