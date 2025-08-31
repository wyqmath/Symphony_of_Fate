# ==============================================================================
# SECTION 1: CONFIGURATION AND IMPORTS
# ==============================================================================
import os
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- New Imports for ESM-2 ---
from transformers import AutoTokenizer, EsmModel

# --- New Import for GFLOPs Calculation ---
from thop import profile

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

# --- Model and Tokenizer ---
# *****************************************************************************
# ** 关键点 **: 您可以在这里切换不同大小的ESM-2模型
# "facebook/esm2_t6_8M_UR50D" -> 800万参数 (用于快速测试)
#  esm2_t12_35M_UR50D
# "facebook/esm2_t30_150M_UR50D" -> 1.5亿参数 (中等大小)
# "facebook/esm2_t33_650M_UR50D" -> 6.5亿参数 (大型，用于最终报告)
# *****************************************************************************
MODEL_NAME = "facebook/esm2_t6_8M_UR50D" 

# --- Hyperparameters for Fine-tuning ---
LEARNING_RATE = 1e-5 
BATCH_SIZE = 16
NUM_EPOCHS = 100
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENT_NAME = f"ESM2_FineTune_{MODEL_NAME.split('/')[-1]}"


# ==============================================================================
# SECTION 2: FASTA DATA LOADING AND TOKENIZATION FOR ESM-2 (FIXED)
# ==============================================================================
def load_esm_data(train_dir, test_dir, tokenizer):
    print(f"--- Loading FASTA Data for ESM-2 ({MODEL_NAME}) ---")
    
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
                seq_with_spaces = " ".join(list(str(record.seq).upper()))
                sequences.append(seq_with_spaces)
                labels.append(label)
                sequences_in_file += 1
        return sequences, labels

    train_seqs, y_train = _load_sequences(train_dir, limit_per_file=800)
    test_seqs, y_test = _load_sequences(test_dir, limit_per_file=200)

    if not train_seqs:
        print("Error: No training sequences found. Aborting.")
        return None, None, None, None

    print("Tokenizing sequences...")
    # --- 关键修复：明确指定最大长度来避免内存溢出 ---
    MAX_LENGTH = 1024 

    X_train_encodings = tokenizer(
        train_seqs, 
        truncation='longest_first', 
        padding='max_length', 
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    X_test_encodings = tokenizer(
        test_seqs, 
        truncation='longest_first', 
        padding='max_length', 
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    print(f"Data processing complete. All sequences truncated/padded to {MAX_LENGTH}.")
    return X_train_encodings, np.array(y_train), X_test_encodings, np.array(y_test)

# ==============================================================================
# SECTION 3: PYTORCH DATASET AND MODEL DEFINITION
# ==============================================================================

class ESMProteinDataset(Dataset):
    """Custom PyTorch Dataset for ESM-2 tokenized sequences."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.from_numpy(labels).long()

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

class ESMClassifier(nn.Module):
    """ESM-2 based classifier."""
    def __init__(self, model_name, num_classes):
        super(ESMClassifier, self).__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self.hidden_size = self.esm.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_representation)
        return logits

# ==============================================================================
# SECTION 4: TRAINING AND EVALUATION FUNCTIONS
# ==============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device):
    print("\n--- Starting Model Training with Early Stopping ---")
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [T]", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = correct_predictions / total_samples

        model.eval()
        val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = correct_val_predictions / total_val_samples
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
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
    print("\n--- Evaluating Model on Test Set ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Using model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    X_train_full_enc, y_train_full, X_test_enc, y_test = load_esm_data(fasta_train_dir, fasta_test_dir, tokenizer)

    if X_train_full_enc is None:
        print("Data loading failed. Exiting.")
    else:
        train_indices, val_indices = train_test_split(
            range(len(y_train_full)), test_size=0.2, random_state=42, stratify=y_train_full
        )
        
        X_train_enc = {key: val[train_indices] for key, val in X_train_full_enc.items()}
        X_val_enc = {key: val[val_indices] for key, val in X_train_full_enc.items()}
        y_train, y_val = y_train_full[train_indices], y_train_full[val_indices]

        print("\nSplit full training data into training (90%) and validation (10%) sets.")

        train_dataset = ESMProteinDataset(X_train_enc, y_train)
        val_dataset = ESMProteinDataset(X_val_enc, y_val)
        test_dataset = ESMProteinDataset(X_test_enc, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("\nDataLoaders created successfully.")

        model = ESMClassifier(model_name=MODEL_NAME, num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print("\nModel, Loss Function, and Optimizer initialized.")

        start_time = time.time()
        best_model_state_dict = train_model(
            model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, DEVICE
        )
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")

        if best_model_state_dict:
            print("\nLoading best model state for final evaluation...")
            model.load_state_dict(best_model_state_dict)
        else:
            print("\nWarning: No best model state saved. Evaluating the last state.")
            
        y_true, y_pred = evaluate_model(model, test_loader, DEVICE)
        
        accuracy = accuracy_score(y_true, y_pred)
        print("\n" + "="*30)
        print(f"FINAL RESULTS for {EXPERIMENT_NAME}")
        print("="*30)
        print(f'Test Set Accuracy: {accuracy * 100:.2f}%')
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

        print('Plotting and saving Confusion Matrix...')
        fig, ax = plt.subplots(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(cmap='viridis', values_format='d', ax=ax, xticks_rotation='vertical')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(f'Confusion Matrix ({EXPERIMENT_NAME})', fontsize=16, pad=20)
        fig.tight_layout()
        filename = f"Confusion_Matrix_{EXPERIMENT_NAME}.png"
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved confusion matrix plot to: {filename}")
        plt.close(fig)

        # ==============================================================================
        # SECTION 6: COMPUTATIONAL COST ANALYSIS
        # ==============================================================================
        print("\n" + "="*30)
        print("COMPUTATIONAL COST ANALYSIS")
        print("="*30)
        
        model.eval()
        
        dummy_batch = next(iter(test_loader))
        dummy_input_ids = dummy_batch['input_ids'][0:1].to(DEVICE) # Take a single sample
        dummy_attention_mask = dummy_batch['attention_mask'][0:1].to(DEVICE)
        
        # 1. GFLOPs and Parameters
        macs, params = profile(model, inputs=(dummy_input_ids, dummy_attention_mask), verbose=False)
        gflops = (macs * 2) / 1e9
        print(f"Model Parameters: {params / 1e6:.2f} M")
        print(f"GFLOPs (per sequence of length {dummy_input_ids.shape[1]}): {gflops:.2f} G")
        
        # 2. Peak GPU Memory
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
        peak_memory_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
        print(f"Inference Peak GPU Memory: {peak_memory_gb:.2f} GB")
        
        # 3. Inference Throughput
        num_sequences_for_throughput_test = 500
        sequences_processed = 0
        
        # Warm-up
        for _ in range(5):
             with torch.no_grad():
                batch = next(iter(test_loader))
                _ = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        test_iter = iter(test_loader)
        while sequences_processed < num_sequences_for_throughput_test:
            try:
                batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader) # Reset iterator if dataset is exhausted
                batch = next(test_iter)

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            sequences_processed += input_ids.size(0)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = sequences_processed / total_time
        print(f"Inference Throughput: {throughput:.2f} sequences/second")