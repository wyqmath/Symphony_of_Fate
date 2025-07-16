import os
import cv2
import numpy as np
import time
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
from torch.nn.utils.rnn import pad_sequence

# ==============================================================================
# SECTION 0: CONFIGURATION (Preserved from your script)
# ==============================================================================
# --- Matplotlib Configuration ---
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

# --- Constants and Global Config (Preserved from your script) ---
CLASS_NAMES = [
    'Enzyme', 'Structural', 'Transport', 'Storage',
    'Signalling', 'Receptor', 'Gene Regulatory',
    'Immune', 'Chaperone'
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Directories (Preserved from your script) ---
spectrogram_train_dir = 'train_image'
spectrogram_test_dir = 'test_image'

# --- Model and Training Parameters (Adapted from script 1) ---
MODEL_HUB_ID = 'convnext_base.fb_in22k'
N_MFCC = 60
BATCH_SIZE = 16 # Adjusted based on your original script's dataloader
NUM_WORKERS = 4
IMAGE_MODEL_WEIGHTS_PATH = 'best_ConvNeXt_ImageOnly_Baseline.pth'
FINAL_MODEL_FILENAME_TAG = 'ConvNeXt_Fusion_AudioAttention'


# ==============================================================================
# SECTION 1: DATA LOADING & UTILITIES (Upgraded to handle sequences)
# ==============================================================================
def extract_mfcc_sequence(spectrogram_np):
    """Extracts a sequence of MFCCs from a spectrogram image."""
    power_spectrogram = np.abs(spectrogram_np.astype(float))**2
    mfccs = librosa.feature.mfcc(S=power_spectrogram, sr=22050, n_mfcc=N_MFCC)
    return mfccs.T # Transpose to get (time_steps, n_mfcc)

class SpectrogramDataset(Dataset):
    """Dataset to load spectrogram images and extract MFCC sequences."""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image_np is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        
        # KEY CHANGE: Extract the full MFCC sequence
        mfcc_sequence = extract_mfcc_sequence(image_np)
        label = self.labels[idx]
        
        if self.transform:
            image_pil = transforms.ToPILImage()(image_np)
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = torch.from_numpy(image_np).float().unsqueeze(0)
            
        mfcc_tensor = torch.tensor(mfcc_sequence, dtype=torch.float)
        return image_tensor, mfcc_tensor, label

def collate_mfcc(batch):
    """Pads MFCC sequences to the same length in a batch for RNN processing."""
    images, mfccs, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    mfccs_padded = pad_sequence(mfccs, batch_first=True, padding_value=0.0)
    return images, mfccs_padded, labels

class SpecAugment(nn.Module):
    """Applies frequency and time masking to spectrograms."""
    def __init__(self, freq_mask_param, time_mask_param, num_freq_masks=1, num_time_masks=1):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spec_tensor):
        aug_spec = spec_tensor.clone()
        _, num_freq_bins, num_time_steps = aug_spec.shape
        mask_value = aug_spec.mean()
        for _ in range(self.num_freq_masks):
            f = int(np.random.uniform(0, self.freq_mask_param))
            f0 = int(np.random.uniform(0, num_freq_bins - f))
            if f > 0: aug_spec[:, f0:f0 + f, :] = mask_value
        for _ in range(self.num_time_masks):
            t = int(np.random.uniform(0, self.time_mask_param))
            t0 = int(np.random.uniform(0, num_time_steps - t))
            if t > 0: aug_spec[:, :, t0:t0 + t] = mask_value
        return aug_spec

# ==============================================================================
# SECTION 2: MODEL ARCHITECTURES (Upgraded to two-phase models)
# ==============================================================================

# --- Phase 1 Model ---
class ImageOnlyModel(nn.Module):
    """A simple image classification model for generating baseline weights."""
    def __init__(self, model_name, num_classes, pretrained=True, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=1, num_classes=0, drop_path_rate=drop_path_rate
        )
        num_cnn_features = self.backbone.num_features
        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(num_cnn_features),
            nn.Linear(num_cnn_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, image):
        features = self.backbone(image)
        output = self.classifier_head(features)
        return output

# --- Phase 2 Model Components ---
class AudioAttention(nn.Module):
    """Attention mechanism to weigh the importance of different audio time steps."""
    def __init__(self, input_dim):
        super().__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
    def forward(self, gru_outputs):
        attention_scores = self.attention_layer(gru_outputs).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), gru_outputs).squeeze(1)
        return context_vector

class AttentionFusionModel(nn.Module):
    """The final fusion model with an attentive GRU for audio processing."""
    def __init__(self, model_name, num_classes, n_mfcc_features, gru_hidden_size=128, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=0, drop_path_rate=drop_path_rate)
        num_cnn_features = self.backbone.num_features
        gru_output_features = gru_hidden_size * 2
        self.mfcc_gru = nn.GRU(
            input_size=n_mfcc_features, hidden_size=gru_hidden_size, 
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.audio_attention = AudioAttention(gru_output_features)
        total_features = num_cnn_features + gru_output_features
        print(f"CNN features: {num_cnn_features}, Attentive Audio features: {gru_output_features}, Total: {total_features}")
        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(total_features),
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, image, mfcc_sequence):
        cnn_features = self.backbone(image)
        gru_outputs, _ = self.mfcc_gru(mfcc_sequence)
        audio_features = self.audio_attention(gru_outputs)
        combined_features = torch.cat([cnn_features, audio_features], dim=1)
        output = self.classifier_head(combined_features)
        return output

# ==============================================================================
# SECTION 3: TRAINING & EVALUATION FUNCTIONS (Upgraded for two-phase flow)
# ==============================================================================

def run_phase1_training(train_loader, val_loader, num_classes):
    """Trains the ImageOnlyModel to generate baseline weights."""
    print("\n--- Starting Phase 1: Image-Only Baseline Training ---")
    model = ImageOnlyModel(model_name=MODEL_HUB_ID, num_classes=num_classes, pretrained=True, drop_path_rate=0.2)
    model.to(DEVICE)
    
    finetune_params = [p for p in model.backbone.parameters()]
    head_params = list(model.classifier_head.parameters())
    param_groups = [{'params': finetune_params, 'lr': 1e-4}, {'params': head_params, 'lr': 1e-3}]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 15
    
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/100 [Train]", leave=False)
        for images, _, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        train_loss_epoch = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss_epoch = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[-1]['lr']
        print(f"Phase 1 - Epoch {epoch+1}/100 | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Head LR: {current_lr:.6f}")
        
        if scheduler: scheduler.step()
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), IMAGE_MODEL_WEIGHTS_PATH)
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving weights to {IMAGE_MODEL_WEIGHTS_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered for Phase 1.")
            break
    print(f"\n--- Phase 1 Training Complete. Weights saved to {IMAGE_MODEL_WEIGHTS_PATH} ---")


def run_phase2_training_and_eval(train_loader, val_loader, test_loader, class_labels):
    """Trains and evaluates the final AttentionFusionModel."""
    print("\n--- Starting Phase 2: Attention Fusion Model Training ---")
    num_classes = len(class_labels)
    model = AttentionFusionModel(model_name=MODEL_HUB_ID, num_classes=num_classes, n_mfcc_features=N_MFCC)
    
    print(f"\nLoading pre-trained backbone weights from: {IMAGE_MODEL_WEIGHTS_PATH}")
    image_only_state_dict = torch.load(IMAGE_MODEL_WEIGHTS_PATH)
    backbone_weights = {k: v for k, v in image_only_state_dict.items() if k.startswith('backbone.')}
    model.load_state_dict(backbone_weights, strict=False)
    print("Successfully loaded backbone weights into the fusion model.")
    model.to(DEVICE)

    head_params = list(model.mfcc_gru.parameters()) + list(model.audio_attention.parameters()) + list(model.classifier_head.parameters())
    optimizer = optim.AdamW([{'params': head_params, 'lr': 1e-4}], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 20
    freeze_epochs = 5
    model_save_path = f'best_{FINAL_MODEL_FILENAME_TAG}.pth'

    for epoch in range(100):
        if epoch == 0:
            print(f"--- Stage A: Training Head Only (Epochs 1-{freeze_epochs}) ---")
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif epoch == freeze_epochs:
            print(f"\n--- Stage B: Unfreezing Backbone (Epoch {freeze_epochs+1}-onwards) ---")
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("Adding backbone parameters to optimizer.")
            optimizer.add_param_group({'params': model.backbone.parameters(), 'lr': 1e-5})

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Phase 2 - Epoch {epoch+1}/100 [Train]", leave=False)
        for images, mfccs, labels in pbar:
            images, mfccs, labels = images.to(DEVICE), mfccs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, mfccs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        train_loss_epoch = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, mfccs, labels in val_loader:
                images, mfccs, labels = images.to(DEVICE), mfccs.to(DEVICE), labels.to(DEVICE)
                outputs = model(images, mfccs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss_epoch = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Phase 2 - Epoch {epoch+1}/100 | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Head LR: {current_lr:.6f}")
        
        if scheduler: scheduler.step()
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving model to {model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered for Phase 2.")
            break
            
    print(f"\nTraining complete. Loading best model weights (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    
    print(f"\n--- Running Final Evaluation for: {FINAL_MODEL_FILENAME_TAG} ---")
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, mfccs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images, mfccs = images.to(DEVICE), mfccs.to(DEVICE)
            outputs = model(images, mfccs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'\nTest Set Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_labels, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap='Greens', values_format='d', ax=ax, xticks_rotation='vertical')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f'Confusion Matrix ({FINAL_MODEL_FILENAME_TAG})', fontsize=16, pad=20)
    fig.tight_layout()
    filename = f"Confusion_Matrix_{FINAL_MODEL_FILENAME_TAG}.png"
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\nSaved confusion matrix plot to: {filename}")
    plt.close(fig)


# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK (Adapted to your data structure)
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Load Data using your specified directory structure ---
    print(f"\n--- Using Pre-trained Model from Timm: {MODEL_HUB_ID} ---")
    print("\n--- Loading file paths and labels from pre-split directories ---")
    
    def get_paths_and_labels(folder):
        paths, labels = [], []
        if not os.path.exists(folder):
            print(f"Error: Directory not found: {folder}")
            return paths, labels
        
        for label_folder in sorted(os.listdir(folder)):
            label_path = os.path.join(folder, label_folder)
            if os.path.isdir(label_path) and label_folder.isdigit():
                for filename in sorted(os.listdir(label_path)):
                    full_path = os.path.join(label_path, filename)
                    if os.path.isfile(full_path) and not filename.startswith('.'):
                        paths.append(full_path)
                        labels.append(int(label_folder) - 1)
        return paths, labels

    all_train_paths, all_train_labels = get_paths_and_labels(spectrogram_train_dir)
    test_paths, test_labels = get_paths_and_labels(spectrogram_test_dir)

    if not all_train_paths or not test_paths:
        print("Error: Training or testing images not found. Aborting.")
    else:
        print(f"Found {len(all_train_paths)} total training images. Splitting into Train/Validation sets.")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_train_paths, all_train_labels, test_size=0.2, random_state=42, stratify=all_train_labels
        )
        print(f"--> New Train Set Size: {len(train_paths)}")
        print(f"--> Validation Set Size: {len(val_paths)}")
        print(f"Found {len(test_paths)} testing images.")
        
        # --- 2. Configure Data Transforms ---
        print(f"\n--- Configuring data transforms for {MODEL_HUB_ID} ---")
        data_config = timm.data.resolve_data_config({}, model=MODEL_HUB_ID)
        input_size = data_config['input_size'][1]
        norm_mean, norm_std = (sum(data_config['mean']) / 3,), (sum(data_config['std']) / 3,)
        print(f"Input size: {input_size}x{input_size}, Mean: {norm_mean}, Std: {norm_std}")

        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            SpecAugment(freq_mask_param=25, time_mask_param=50, num_freq_masks=2, num_time_masks=2),
            transforms.Normalize(norm_mean, norm_std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)
        ])

        train_dataset = SpectrogramDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = SpectrogramDataset(val_paths, val_labels, transform=val_test_transform)
        test_dataset = SpectrogramDataset(test_paths, test_labels, transform=val_test_transform)

        # --- 3. Execute Two-Phase Training Pipeline ---
        
        # Step 1: Check for and run Phase 1 if necessary
        if not os.path.exists(IMAGE_MODEL_WEIGHTS_PATH):
            # For Phase 1, MFCCs are ignored, so no collate_fn is needed
            phase1_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            phase1_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            run_phase1_training(phase1_train_loader, phase1_val_loader, NUM_CLASSES)
        else:
            print(f"Found existing baseline weights at '{IMAGE_MODEL_WEIGHTS_PATH}'. Skipping Phase 1 training.")

        # Step 2: Run Phase 2 using the baseline weights
        # For Phase 2, collate_fn is required for MFCC padding
        phase2_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_mfcc)
        phase2_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_mfcc)
        phase2_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_mfcc)
        run_phase2_training_and_eval(phase2_train_loader, phase2_val_loader, phase2_test_loader, CLASS_NAMES)