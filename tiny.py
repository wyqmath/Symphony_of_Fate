import os
import cv2
import numpy as np
import time
import sys
import copy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

# = a============================================================================
# SECTION 0: CONFIGURATION
# ==============================================================================
matplotlib.use('Agg')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Core Parameters ---
TRAIN_DATA_DIR = 'train_image'
TEST_DATA_DIR = 'test_image'
# [MODIFIED] Switched to a smaller backbone to combat severe overfitting. This is the most critical change.
MODEL_HUB_ID = 'convnext_tiny.in12k' 
N_MFCC = 60
BATCH_SIZE = 32
NUM_WORKERS = 128

# --- File Paths & Model Naming ---
# [MODIFIED] Updated tag to reflect the new, smaller backbone
FINAL_MODEL_FILENAME_TAG = 'ConvNeXt_Tiny_MFCC_GatedFusion' 
IMAGE_MODEL_WEIGHTS_PATH = f'stage1_image_expert_weights_{MODEL_HUB_ID}.pth' # Path is now specific to the tiny model

# --- Class Names ---
CLASS_NAMES = ['Enzyme', 'Structural', 'Transport', 'Storage', 'Signalling', 'Receptor', 'Gene Regulatory', 'Immune', 'Chaperone']
NUM_CLASSES = len(CLASS_NAMES)

# --- Training Hyperparameters ---
PHASE1_EPOCHS = 100
PHASE1_PATIENCE = 20
FUSION_EPOCHS = 150
FUSION_PATIENCE = 10

# ==============================================================================
# SECTION 1: DATA LOADING & UTILITIES
# ==============================================================================

def extract_mfcc_mean(spectrogram_np, n_mfcc=N_MFCC):
    if spectrogram_np is None or spectrogram_np.size == 0: return np.zeros((n_mfcc,))
    try:
        power_spectrogram = np.abs(spectrogram_np.astype(float))**2
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(power_spectrogram), sr=22050, n_mfcc=n_mfcc)
        return np.mean(mfccs, axis=1)
    except Exception: return np.zeros((n_mfcc,))

class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths, self.labels, self.transform = file_paths, labels, transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image_np is None: return None
            mfcc_vector = extract_mfcc_mean(image_np)
            label = self.labels[idx]
            image_pil = transforms.ToPILImage()(image_np)
            image_tensor = self.transform(image_pil)
            mfcc_tensor = torch.tensor(mfcc_vector, dtype=torch.float)
            return image_tensor, mfcc_tensor, label
        except Exception: return None

def filter_none_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# ==============================================================================
# SECTION 2: MODEL ARCHITECTURES
# ==============================================================================

class ImageOnlyModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=1, num_classes=0, drop_path_rate=drop_path_rate)
        num_cnn_features = self.backbone.num_features
        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(num_cnn_features),
            nn.Linear(num_cnn_features, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.6), # [MODIFIED] Increased dropout for more regularization
            nn.Linear(512, num_classes)
        )
    def forward(self, image): return self.classifier_head(self.backbone(image))

class GatedFusionModel(nn.Module):
    def __init__(self, model_name, num_classes, n_mfcc_features, pretrained=True, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=1, num_classes=0, drop_path_rate=drop_path_rate)
        num_cnn_features = self.backbone.num_features
        total_features = num_cnn_features + n_mfcc_features
        print(f"CNN backbone features: {num_cnn_features}")
        print(f"MFCC vector features: {n_mfcc_features}")
        print(f"Total combined features: {total_features}")
        self.attention_gate = nn.Sequential(
            nn.Linear(total_features, total_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_features // 4, total_features),
            nn.Sigmoid()
        )
        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(total_features),
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6), # [MODIFIED] Increased dropout for more regularization
            nn.Linear(512, num_classes)
        )
    def forward(self, image, mfcc_vector):
        cnn_features = self.backbone(image)
        combined_features = torch.cat([cnn_features, mfcc_vector], dim=1)
        attention_weights = self.attention_gate(combined_features)
        gated_features = combined_features * attention_weights
        return self.classifier_head(gated_features)

# ==============================================================================
# SECTION 3: TRAINING & EVALUATION FUNCTIONS
# ==============================================================================

def run_stage1_image_training(train_loader, val_loader, num_classes, epochs, patience):
    print("\n--- Starting Stage 1: Image Expert Fine-tuning ---")
    model = ImageOnlyModel(model_name=MODEL_HUB_ID, num_classes=num_classes, pretrained=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_loss = float('inf'); epochs_no_improve = 0
    for epoch in range(epochs):
        model.train(); running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, _, labels in pbar:
            if images.nelement() == 0: continue
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss_epoch = running_loss / len(train_loader.dataset)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for images, _, labels in val_loader:
                if images.nelement() == 0: continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images); loss = criterion(outputs, labels); val_loss += loss.item() * images.size(0)
        val_loss_epoch = val_loss / len(val_loader.dataset)
        print(f"Stage 1 - Epoch {epoch+1}/{epochs} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}")
        scheduler.step(val_loss_epoch)
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), IMAGE_MODEL_WEIGHTS_PATH)
            print(f"Val loss improved. Saving image model to {IMAGE_MODEL_WEIGHTS_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered for Stage 1."); break

def run_fusion_training(train_loader, val_loader, test_loader, class_labels, epochs, patience):
    print(f"\n--- Starting End-to-End Fusion Training ({FINAL_MODEL_FILENAME_TAG}) ---")
    model = GatedFusionModel(model_name=MODEL_HUB_ID, num_classes=NUM_CLASSES, n_mfcc_features=N_MFCC, pretrained=False)
    print(f"Loading Stage 1 Image Expert weights from: {IMAGE_MODEL_WEIGHTS_PATH}")
    image_expert_state_dict = torch.load(IMAGE_MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    backbone_weights = {k.replace('backbone.', ''): v for k, v in image_expert_state_dict.items() if k.startswith('backbone.')}
    model.backbone.load_state_dict(backbone_weights)
    print("-> Image backbone weights loaded successfully.")
    model.to(DEVICE)
    
    # [MODIFIED] Reduced head LR to slow down fitting and prevent overfitting
    base_lr = 2e-5; head_lr = 1e-4 
    head_params = list(model.attention_gate.parameters()) + list(model.classifier_head.parameters())
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': base_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=1e-2)
    print(f"Optimizer: AdamW with differential LRs (Backbone: {base_lr}, Head: {head_lr})")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_loss = float('inf'); epochs_no_improve = 0
    model_save_path = f'best_{FINAL_MODEL_FILENAME_TAG}.pth'
    for epoch in range(epochs):
        model.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Fusion Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, mfccs, labels in pbar:
            if images.nelement() == 0: continue
            images, mfccs, labels = images.to(DEVICE), mfccs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); outputs = model(images, mfccs); loss = criterion(outputs, labels); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = correct_predictions / total_samples
        model.eval()
        val_loss, correct_val_predictions, total_val_samples = 0.0, 0, 0
        with torch.no_grad():
            for images, mfccs, labels in val_loader:
                if images.nelement() == 0: continue
                images, mfccs, labels = images.to(DEVICE), mfccs.to(DEVICE), labels.to(DEVICE)
                outputs = model(images, mfccs); loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = correct_val_predictions / total_val_samples
        lr_backbone = optimizer.param_groups[0]['lr']; lr_head = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f} | LRs: {lr_head:.6f}/{lr_backbone:.6f}")
        scheduler.step()
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Val loss improved to {best_val_loss:.4f}. Saving model to {model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs."); break
    print(f"\nTraining complete. Loading best model from {model_save_path} for final evaluation.")
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    evaluate_with_tta(model, test_loader, class_labels)

def evaluate_with_tta(model, test_loader, class_labels):
    print("\n--- Evaluating with Test Time Augmentation (TTA) ---")
    model.eval(); all_labels, all_preds = [], []
    with torch.no_grad():
        for images, mfccs, labels in tqdm(test_loader, desc="Evaluating on Test Set with TTA"):
            if images.nelement() == 0: continue
            images_orig, images_flipped = images.to(DEVICE), torch.flip(images, [3]).to(DEVICE)
            mfccs = mfccs.to(DEVICE)
            outputs_orig, outputs_flipped = model(images_orig, mfccs), model(images_flipped, mfccs)
            avg_outputs = (torch.softmax(outputs_orig, dim=1) + torch.softmax(outputs_flipped, dim=1)) / 2
            _, predicted = torch.max(avg_outputs, 1)
            all_labels.extend(labels.numpy()); all_preds.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'\nFinal TTA Test Set Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report (TTA):"); print(classification_report(all_labels, all_preds, target_names=class_labels, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds); disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(12, 10)); disp.plot(cmap='Greens', values_format='d', ax=ax, xticks_rotation='vertical')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f'Confusion Matrix ({FINAL_MODEL_FILENAME_TAG} with TTA)', fontsize=16, pad=20); fig.tight_layout()
    filename = f"Confusion_Matrix_{FINAL_MODEL_FILENAME_TAG}_TTA.png"; fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\nSaved TTA confusion matrix to: {filename}"); plt.close(fig)

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    def load_data_from_dir(directory, max_files_per_class=None):
        filepaths, labels = [], []
        print(f"Loading data from: {directory}")
        if not os.path.isdir(directory):
            print(f"Error: Directory not found: {directory}"); return filepaths, labels
        for class_name_str in sorted(os.listdir(directory)):
            class_dir = os.path.join(directory, class_name_str)
            if not os.path.isdir(class_dir) or not class_name_str.isdigit(): continue
            label = int(class_name_str) - 1
            if not (0 <= label < NUM_CLASSES):
                print(f"Warning: Skipping folder '{class_name_str}' with out-of-range label {label}.")
                continue
            files_in_class = sorted(os.listdir(class_dir))
            files_added = 0
            for fname in files_in_class:
                if max_files_per_class is not None and files_added >= max_files_per_class: break
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath):
                    filepaths.append(fpath); labels.append(label); files_added += 1
        print(f"  -> Loaded a total of {len(filepaths)} files.")
        return filepaths, labels

    train_full_files, train_full_labels = load_data_from_dir(TRAIN_DATA_DIR)
    test_files, test_labels = load_data_from_dir(TEST_DATA_DIR, max_files_per_class=200)
    if not train_full_files: sys.exit("Error: No training data found. Exiting.")
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_full_files, train_full_labels, test_size=0.15, random_state=42, stratify=train_full_labels)
    print("\n--- Data Split Summary ---")
    print(f"Training: {len(train_files)} | Validation: {len(val_files)} | Test: {len(test_files)}")
    print("--------------------------\n")

    data_config = timm.data.resolve_model_data_config(MODEL_HUB_ID)
    input_size = data_config['input_size'][1]
    norm_mean, norm_std = (data_config['mean'][0],), (data_config['std'][0],)
    print(f"Using model-specific normalization: mean={norm_mean}, std={norm_std}, input_size={input_size}")

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    train_dataset = SpectrogramDataset(train_files, train_labels, transform=train_transform)
    val_dataset = SpectrogramDataset(val_files, val_labels, transform=val_test_transform)
    test_dataset = SpectrogramDataset(test_files, test_labels, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=filter_none_collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, collate_fn=filter_none_collate, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, collate_fn=filter_none_collate, pin_memory=True)

    # Since we changed the model, we MUST re-run Stage 1 training for the new backbone
    # Forcing re-training by checking for the new, specific weights file
    if not os.path.exists(IMAGE_MODEL_WEIGHTS_PATH):
        print(f"Weights for {MODEL_HUB_ID} not found. Starting Stage 1 training...")
        run_stage1_image_training(train_loader, val_loader, NUM_CLASSES, epochs=PHASE1_EPOCHS, patience=PHASE1_PATIENCE)
    else:
        print(f"\nFound Stage 1 weights at '{IMAGE_MODEL_WEIGHTS_PATH}'. Skipping image expert fine-tuning.")
    
    run_fusion_training(
        train_loader, val_loader, test_loader, CLASS_NAMES,
        epochs=FUSION_EPOCHS, patience=FUSION_PATIENCE)

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()
    print("\n--- All processes completed successfully! ---")