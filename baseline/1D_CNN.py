# ==============================================================================
# SECTION 1: CONFIGURATION AND IMPORTS
# ==============================================================================
import os
import random
import time
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
# 新增导入，用于划分验证集
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Matplotlib Configuration ---
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

# Model Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 100  
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENT_NAME = "1D_Hydrophobicity_CNN"


# --- Physicochemical Properties (保持不变) ---
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2, 'U': 0.0, 'X': 0.0, 'B': -3.5, 'Z': -3.5, 'J': 3.8
}
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0}, 'R': {'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3}, 'N': {'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3}, 'D': {'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3}, 'C': {'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1}, 'Q': {'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2}, 'E': {'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2}, 'G': {'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3}, 'H': {'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3}, 'I': {'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3}, 'L': {'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1}, 'K': {'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2}, 'M': {'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1}, 'F': {'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1}, 'P': {'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2}, 'S': {'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2}, 'T': {'T': 5, 'W': -2, 'Y': -2, 'V': 0}, 'W': {'W': 11, 'Y': 2, 'V': -3}, 'Y': {'Y': 7, 'V': -1}, 'V': {'V': 4}
}

# ==============================================================================
# SECTION 2: FASTA DATA LOADING AND PROCESSING (保持不变)
# ==============================================================================
def _conservative_mutation(sequence, num_mutations=2):
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
    original_len = len(sequence)
    crop_len = int(original_len * crop_ratio)
    if crop_len >= original_len: return sequence
    start_index = random.randint(0, original_len - crop_len)
    return sequence[start_index : start_index + crop_len]

def load_1d_signal_data(train_dir, test_dir):
    print("--- Loading and Processing FASTA Data for 1D CNN ---")
    def _load_sequences(fasta_dir, limit_per_file):
        sequences, labels = [], []
        if not os.path.exists(fasta_dir):
            print(f"Warning: Directory not found: {fasta_dir}")
            return sequences, labels
        for filename in tqdm(os.listdir(fasta_dir), desc=f"Reading files from {os.path.basename(fasta_dir)}"):
            if not filename.endswith(('.fasta', '.fa')): continue
            try: label = int(os.path.splitext(filename)[0]) - 1
            except ValueError: continue
            filepath = os.path.join(fasta_dir, filename)
            sequences_in_file = 0
            for record in SeqIO.parse(filepath, "fasta"):
                if sequences_in_file >= limit_per_file: break
                seq = str(record.seq).upper()
                if 'X' in seq: continue
                sequences.append(seq)
                labels.append(label)
                sequences_in_file += 1
        return sequences, labels
    train_seqs, y_train_orig = _load_sequences(train_dir, limit_per_file=800)
    test_seqs, y_test = _load_sequences(test_dir, limit_per_file=200)
    if not train_seqs:
        print("Error: No training sequences found. Aborting.")
        return [np.array([])] * 4
    print("Augmenting training sequences...")
    aug_train_seqs, aug_train_labels = [], []
    for seq, label in tqdm(zip(train_seqs, y_train_orig), total=len(train_seqs), desc="Augmenting"):
        aug_train_seqs.append(seq)
        aug_train_labels.append(label)
        aug_train_seqs.append(_conservative_mutation(seq))
        aug_train_labels.append(label)
        aug_train_seqs.append(_random_cropping(seq))
        aug_train_labels.append(label)
    # 修正数据泄露问题
    max_len = max(len(s) for s in aug_train_seqs) if aug_train_seqs else 0
    print(f"Max protein length (determined from training set only) set to: {max_len}")
    def _sequences_to_1d_signals(sequences, apply_noise=False, noise_level=0.05):
        signals = []
        for seq in tqdm(sequences, desc="Converting to 1D signals"):
            hydro_signal = np.array([KYTE_DOOLITTLE.get(aa, 0) for aa in seq])
            if apply_noise:
                noise = np.random.normal(0, noise_level, hydro_signal.shape)
                hydro_signal += noise
            if len(hydro_signal) > max_len:
                hydro_signal = hydro_signal[:max_len]
            padded_signal = np.pad(hydro_signal, (0, max_len - len(hydro_signal)), 'constant')
            signals.append(padded_signal)
        return np.array(signals, dtype=np.float32)
    X_train = _sequences_to_1d_signals(aug_train_seqs, apply_noise=True)
    X_test = _sequences_to_1d_signals(test_seqs, apply_noise=False)
    print(f"Data processing complete. Shapes:")
    print(f"X_train (full): {X_train.shape}, y_train (full): {len(aug_train_labels)}")
    print(f"X_test: {X_test.shape}, y_test: {len(y_test)}")
    return X_train, np.array(aug_train_labels), X_test, np.array(y_test), max_len

# ==============================================================================
# SECTION 3: PYTORCH DATASET AND MODEL DEFINITION
# ==============================================================================

class ProteinDataset(Dataset):
    """Custom PyTorch Dataset for protein signals."""
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).unsqueeze(1)
        self.labels = torch.from_numpy(labels).long()
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class CNN1D(nn.Module):
    """1D CNN for protein sequence classification with Batch Normalization."""
    def __init__(self, input_length, num_classes):
        super(CNN1D, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3),
            # 修改点：加入批标准化
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            # 修改点：加入批标准化
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # 修改点：加入批标准化
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.flattened_size = self._get_flattened_size(input_length)
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def _get_flattened_size(self, input_length):
        x = torch.zeros(1, 1, input_length)
        x = self.conv_stack(x)
        return x.numel()
    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.fc_stack(x)
        return logits

# ==============================================================================
# SECTION 4: TRAINING AND EVALUATION FUNCTIONS (MODIFIED FOR EARLY STOPPING)
# ==============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device):
    """Trains the PyTorch model with early stopping."""
    print("\n--- Starting Model Training with Early Stopping ---")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [T]", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = correct_predictions / total_samples

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = correct_val_predictions / total_val_samples
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # --- Early Stopping Check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # 保存性能最佳的模型的状态字典
            best_model_state = model.state_dict()
            print(f"Validation loss improved. Saving model state.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    return best_model_state

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test set."""
    print("\n--- Evaluating Model on Test Set ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK (MODIFIED)
# ==============================================================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # --- 1. Load and prepare data ---
    X_train_full, y_train_full, X_test, y_test, max_len = load_1d_signal_data(fasta_train_dir, fasta_test_dir)

    if X_train_full.size == 0:
        print("Data loading failed. Exiting.")
    else:
        # --- 2. Split training data into training and validation sets ---
        # 使用 stratify=y_train_full 确保训练集和验证集中的类别分布与原始数据集一致
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
        print("\nSplit full training data into training (90%) and validation (10%) sets.")
        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

        # --- 3. Create Datasets and DataLoaders ---
        train_dataset = ProteinDataset(X_train, y_train)
        val_dataset = ProteinDataset(X_val, y_val)
        test_dataset = ProteinDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("\nDataLoaders created successfully.")

        # --- 4. Initialize Model, Loss, and Optimizer (with Weight Decay) ---
        model = CNN1D(input_length=max_len, num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        # 修改点：在优化器中加入 weight_decay
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print("\nModel, Loss Function, and Optimizer initialized.")
        print(f"Model Architecture:\n{model}")

        # --- 5. Train the model ---
        start_time = time.time()
        best_model_state_dict = train_model(
            model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, DEVICE
        )
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")

        # --- 6. Load the best model and evaluate ---
        if best_model_state_dict:
            print("\nLoading best model state for final evaluation...")
            model.load_state_dict(best_model_state_dict)
        else:
            print("\nWarning: No best model state saved. Evaluating the last state.")
            
        y_true, y_pred = evaluate_model(model, test_loader, DEVICE)
        
        # --- 7. Report and Visualize Results ---
        accuracy = accuracy_score(y_true, y_pred)
        print("\n" + "="*30)
        print(f"FINAL RESULTS for {EXPERIMENT_NAME}")
        print("="*30)
        print(f'Test Set Accuracy: {accuracy * 100:.2f}%')
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

        # --- Plot and save confusion matrix ---
        print('Plotting and saving Confusion Matrix...')
        fig, ax = plt.subplots(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(cmap='RdPu', values_format='d', ax=ax, xticks_rotation='vertical')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(f'Confusion Matrix ({EXPERIMENT_NAME})', fontsize=16, pad=20)
        fig.tight_layout()
        filename = f"Confusion_Matrix_{EXPERIMENT_NAME}.png"
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved confusion matrix plot to: {filename}")
        plt.close(fig)