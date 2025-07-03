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

# --- NEW: Import librosa for MFCC feature extraction ---
import librosa

# --- PyTorch and Timm Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

# --- Matplotlib Configuration ---
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
# English-friendly font settings
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

# --- Constants and Global Config ---
CLASS_NAMES = [
    'Enzyme', 'Structural', 'Transport', 'Storage',
    'Signalling', 'Receptor', 'Gene Regulatory',
    'Immune', 'Chaperone'
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- NEW: MFCC Configuration ---
N_MFCC = 60  # Number of MFCC coefficients to extract

# --- Data Directories ---
spectrogram_train_dir = 'train_image'
spectrogram_test_dir = 'test_image'

# ==============================================================================
# SECTION 1: FEATURE EXTRACTION & DATALOADER
# ==============================================================================

# --- MODIFIED & FIXED: MFCC extraction function ---
def extract_features_from_spectrogram(spectrogram_np):
    """
    Extracts MFCC features from a spectrogram in numpy format.
    The input is an image of a spectrogram, which we assume represents amplitude.
    """
    # The input is an image of a spectrogram, which we assume represents amplitude.
    # 1. Convert amplitude spectrogram to power spectrogram.
    #    librosa.feature.mfcc's `S` parameter expects a power spectrogram.
    power_spectrogram = np.abs(spectrogram_np.astype(float))**2
    
    # 2. Extract MFCCs from the power spectrogram.
    #    The `S` parameter is used for spectrogram input. `y` is for time-series audio.
    #    Using `S` resolves the dimensional error and the librosa warning.
    mfccs = librosa.feature.mfcc(S=power_spectrogram, sr=22050, n_mfcc=N_MFCC)
    
    # mfccs shape: (n_mfcc, n_frames), where n_frames is the width of the image.
    # 3. Return the mean of MFCCs over all time frames (axis=1) to get a fixed-size feature vector.
    return np.mean(mfccs, axis=1) # Result shape: (n_mfcc,)


# --- MODIFIED: Update Dataset to return both image and MFCC features ---
class SpectrogramDataset(Dataset):
    """
    Custom PyTorch Dataset.
    After modification, __getitem__ returns (image_tensor, mfcc_tensor, label).
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        # Read the image in grayscale format to get a NumPy array
        image_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image_np is None:
            raise FileNotFoundError(f"Could not read or find the image file, or the file is corrupted: {img_path}")

        # --- KEY MODIFICATION: Extract MFCC features from the raw NumPy image before applying transforms ---
        mfcc_features = extract_features_from_spectrogram(image_np)
        
        label = self.labels[idx]
        
        # Apply image transformations (e.g., to PIL Image, resize, augment, to Tensor)
        if self.transform:
            # The transform expects a PIL image, so we convert the numpy array
            image_pil = transforms.ToPILImage()(image_np)
            image_tensor = self.transform(image_pil)
        else:
            # If no transform, at least convert to a Tensor
            image_tensor = torch.from_numpy(image_np).float().unsqueeze(0) # Add channel dimension

        # Convert MFCC features to a Tensor as well
        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float)
        
        return image_tensor, mfcc_tensor, label

# ==============================================================================
# SECTION 2: NEW - DATA AUGMENTATION (SpecAugment)
# ==============================================================================

class SpecAugment(nn.Module):
    """
    SpecAugment: 专为语谱图设计的强大数据增强方法。
    论文: https://arxiv.org/abs/1904.08779
    
    此实现是一个 PyTorch transform，可以轻松添加到 transforms.Compose 管道中。
    它对语谱图张量应用频率和时间遮挡。
    """
    def __init__(self, freq_mask_param, time_mask_param, num_freq_masks=1, num_time_masks=1, replace_with_zero=False):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.replace_with_zero = replace_with_zero

    def forward(self, spec_tensor):
        """
        Args:
            spec_tensor (Tensor): 语谱图张量，形状为 (C, F, T)
                                  C 是通道数, F 是频率轴, T 是时间轴。
        Returns:
            Tensor: 增强后的语谱图张量。
        """
        aug_spec = spec_tensor.clone()
        _, num_freq_bins, num_time_steps = aug_spec.shape

        # 论文建议用语谱图的均值填充，也可以选择用0填充
        mask_value = 0.0 if self.replace_with_zero else aug_spec.mean()

        # 应用频率遮挡 (Frequency Masking)
        for _ in range(self.num_freq_masks):
            f = int(np.random.uniform(0, self.freq_mask_param)) # 遮挡宽度
            f0 = int(np.random.uniform(0, num_freq_bins - f))   # 起始位置
            if f > 0:
                aug_spec[:, f0:f0 + f, :] = mask_value

        # 应用时间遮挡 (Time Masking)
        for _ in range(self.num_time_masks):
            t = int(np.random.uniform(0, self.time_mask_param)) # 遮挡宽度
            t0 = int(np.random.uniform(0, num_time_steps - t))   # 起始位置
            if t > 0:
                aug_spec[:, :, t0:t0 + t] = mask_value
                
        return aug_spec

    def __repr__(self):
        return self.__class__.__name__ + f'(freq_mask_param={self.freq_mask_param}, ' + \
               f'time_mask_param={self.time_mask_param}, ' + \
               f'num_freq_masks={self.num_freq_masks}, ' + \
               f'num_time_masks={self.num_time_masks})'

# ==============================================================================
# SECTION 3: MODIFIED FUSION MODEL WITH ATTENTION
# ==============================================================================

# --- NEW: Define the Fusion Model with an Attention Mechanism ---
class FusionModelWithAttention(nn.Module):
    """
    融合模型，引入了注意力机制。
    该模型将CNN提取的图像特征与MFCC音频特征进行融合。
    注意力机制动态地学习在分类时应该更“相信”哪一部分特征。
    """
    def __init__(self, model_name, num_classes, n_mfcc_features, pretrained=True, drop_path_rate=0.2):
        super().__init__()
        # 1. 加载预训练的CNN骨干网络
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            in_chans=1,          # 输入是单通道灰度图
            num_classes=0,       # 设置为0以移除原始的分类器头
            drop_path_rate=drop_path_rate
        )
        
        num_cnn_features = self.backbone.num_features
        total_features = num_cnn_features + n_mfcc_features
        
        print(f"CNN backbone output features: {num_cnn_features}")
        print(f"MFCC features: {n_mfcc_features}")
        print(f"Total combined features before attention: {total_features}")

        # 2. 定义注意力门 (Attention Gate)
        # 这个小型的全连接网络学习如何为拼接后的特征向量的不同部分分配权重。
        # 它接收拼接后的特征，并输出一个同等大小、值在0到1之间的“门”或“掩码”。
        self.attention_gate = nn.Sequential(
            nn.Linear(total_features, total_features // 4), # 降维以减少参数
            nn.ReLU(inplace=True),
            nn.Linear(total_features // 4, total_features), # 恢复维度
            nn.Sigmoid()                                    # 输出0到1之间的权重
        )
        
        # 3. 定义最终的分类器头
        # 这个头现在处理经过注意力加权后的特征。
        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(total_features),
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, mfcc):
        # 1. 从图像中提取CNN特征
        cnn_features = self.backbone(image)
        
        # 2. 拼接CNN特征和MFCC特征
        combined_features = torch.cat([cnn_features, mfcc], dim=1)
        
        # 3. 生成注意力权重
        # 模型通过attention_gate学习在当前样本中，图像特征和音频特征哪个更重要。
        attention_weights = self.attention_gate(combined_features)
        
        # 4. 应用注意力权重 (Gating)
        # 将原始的拼接特征与学习到的权重进行逐元素相乘，从而对特征进行缩放。
        gated_features = combined_features * attention_weights
        
        # 5. 将加权后的特征送入分类器头进行最终分类
        output = self.classifier_head(gated_features)
        return output

# ==============================================================================
# SECTION 4: CNN TRAINING & EVALUATION (Unchanged, compatible with new model)
# ==============================================================================

def train_cnn_model(model, train_loader, val_loader, optimizer, scheduler, model_save_path, epochs=100, patience=15):
    """Fine-tunes the fusion model."""
    print(f"\n--- Starting Fusion Model Training on {DEVICE} ---")
    print(f"Epochs: {epochs}, Early Stopping Patience: {patience}")
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, mfccs, labels in train_pbar:
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
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validate]", leave=False)
            for images, mfccs, labels in val_pbar:
                images, mfccs, labels = images.to(DEVICE), mfccs.to(DEVICE), labels.to(DEVICE)
                outputs = model(images, mfccs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss_epoch = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | LR: {current_lr:.6f}")
        
        if scheduler:
            scheduler.step()
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving model to {model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break
            
    print(f"\nTraining complete. Loading best model weights (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    return model

def evaluate_end_to_end(model, test_loader, class_labels, model_name):
    """Evaluates the trained fusion model on the test set."""
    print("\n" + "="*80)
    print(f"RUNNING FINAL EVALUATION FOR: {model_name}")
    print("="*80)
    
    model.to(DEVICE)
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, mfccs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images, mfccs = images.to(DEVICE), mfccs.to(DEVICE)
            outputs = model(images, mfccs)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*30)
    print(f"FINAL RESULTS FOR FUSION MODEL: {model_name}")
    print("="*30)
    print(f'Test Set Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_labels, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap='Greens', values_format='d', ax=ax, xticks_rotation='vertical')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f'Confusion Matrix ({model_name})', fontsize=16, pad=20)
    fig.tight_layout()
    filename = f"Confusion_Matrix_{model_name.replace(' ', '_')}.png"
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\nSaved confusion matrix plot to: {filename}")
    plt.close(fig)

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK (Updated for Attention Model)
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Define Model and Load Data ---
    MODEL_HUB_ID = 'convnext_base.fb_in22k'
    # MODIFICATION: Updated filename tag to reflect the new model architecture
    MODEL_FILENAME_TAG = 'ConvNeXt_Base_MFCC_Fusion_Attention'
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
            all_train_paths,
            all_train_labels,
            test_size=0.2,
            random_state=42,
            stratify=all_train_labels
        )
        print(f"--> New Train Set Size: {len(train_paths)}")
        print(f"--> Validation Set Size: {len(val_paths)}")
        print(f"Found {len(test_paths)} testing images.")
        
        # --- 2. Configure Data Transforms ---
        print(f"\n--- Configuring data transforms for {MODEL_HUB_ID} ---")
        temp_model_for_config = timm.create_model(MODEL_HUB_ID, pretrained=True)
        data_config = timm.data.resolve_data_config({}, model=temp_model_for_config)
        input_size = data_config['input_size'][1]
        
        norm_mean = (sum(data_config['mean']) / 3,)
        norm_std = (sum(data_config['std']) / 3,)
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
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

        # --- 3. Create Datasets and DataLoaders ---
        train_dataset = SpectrogramDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = SpectrogramDataset(val_paths, val_labels, transform=val_test_transform)
        test_dataset = SpectrogramDataset(test_paths, test_labels, transform=val_test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        # --- 4. Initialize Fusion Model and Optimizer ---
        # MODIFICATION: Instantiate the new FusionModelWithAttention
        model = FusionModelWithAttention(
            model_name=MODEL_HUB_ID,
            num_classes=NUM_CLASSES,
            n_mfcc_features=N_MFCC,
            pretrained=True,
            drop_path_rate=0.2
        )
        
        # MODIFICATION: Implement differential learning rates for the new model structure
        print("\n--- Setting up differential learning rates for Attention Fusion Model ---")
        base_lr = 2e-5  # Lower learning rate for the pre-trained backbone
        head_lr = 1e-4  # Higher learning rate for the new attention gate and classifier head
        
        # The new head consists of the attention_gate and the classifier_head.
        # Both are new parts and should use the higher learning rate.
        head_params = list(model.attention_gate.parameters()) + list(model.classifier_head.parameters())
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': base_lr},
            {'params': head_params, 'lr': head_lr}
        ]
        
        print(f"Backbone LR: {base_lr}, Attention & Classifier Head LR: {head_lr}")
        
        optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
        total_epochs = 100
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
        
        # --- 5. Train Model ---
        model_save_path = f'best_{MODEL_FILENAME_TAG}.pth'
        
        trained_model = train_cnn_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model_save_path=model_save_path, 
            epochs=total_epochs, 
            patience=15
        )
        
        # --- 6. Final Evaluation on the Test Set ---
        evaluate_end_to_end(
            model=trained_model,
            test_loader=test_loader,
            class_labels=CLASS_NAMES,
            model_name=MODEL_FILENAME_TAG
        )