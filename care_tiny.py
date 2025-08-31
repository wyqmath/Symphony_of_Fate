import os
import cv2
import numpy as np
import time
import copy
import sys
import random
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
# SECTION 0: CONFIGURATION
# ==============================================================================
matplotlib.use('Agg')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Core Parameters ---
DATA_DIR = '/root/autodl-tmp/JCIM/care_data/Enzyme_Classification/enzyme_output/image' # 请确保这是你的数据主目录
MODEL_HUB_ID = 'convnext_tiny.in12k'
N_MFCC = 60
BATCH_SIZE = 32
NUM_WORKERS = 64 # 根据你的机器配置调整

# --- File Paths & Naming (学习自脚本1的规范) ---
FINAL_MODEL_FILENAME_TAG = 'ConvNeXt_Tiny_Fusion_CRNN_Mixup_EC'
IMAGE_MODEL_WEIGHTS_PATH = f'phase1_weights_for_{FINAL_MODEL_FILENAME_TAG}.pth'

# --- 训练超参数 ---
# 阶段一 (纯图像模型) 训练参数
PHASE1_EPOCHS = 100
PHASE1_PATIENCE = 15

# 阶段二 (融合模型) 训练参数
PHASE2_EPOCHS = 200
PHASE2_PATIENCE = 25 # 融合模型更复杂，可以适当增加耐心
PHASE2_FREEZE_EPOCHS = 20 # 在解冻主干网络前，仅训练新模块的轮数

# ==============================================================================
# SECTION 1: DATA LOADING & UTILITIES
# ==============================================================================

def extract_mfcc_sequence(spectrogram_np):
    """从频谱图图像 (NumPy array) 中提取 MFCC 时间序列。"""
    if spectrogram_np is None or spectrogram_np.size == 0:
        return np.zeros((1, N_MFCC))
    power_spectrogram = np.abs(spectrogram_np.astype(float))**2
    mfccs = librosa.feature.mfcc(S=power_spectrogram, sr=22050, n_mfcc=N_MFCC)
    return mfccs.T

class SpectrogramDataset(Dataset):
    """
    自定义数据集，用于加载频谱图并提取其对应的MFCC序列。
    能优雅地处理读取失败或损坏的图像文件。
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image_np is None:
                print(f"警告: 无法读取图像 {img_path}，文件可能已损坏。将跳过此样本。")
                return None
          
            mfcc_sequence = extract_mfcc_sequence(image_np)
            label = self.labels[idx]
          
            image_pil = transforms.ToPILImage()(image_np)
            image_tensor = self.transform(image_pil)
              
            mfcc_tensor = torch.tensor(mfcc_sequence, dtype=torch.float)
            return image_tensor, mfcc_tensor, label
        except Exception as e:
            print(f"警告: 处理文件 {img_path} 时发生错误: {e}。将跳过此样本。")
            return None

def collate_mfcc_and_filter_none(batch):
    """
    自定义 collate_fn:
    - 过滤掉数据集中损坏的样本 (None)。
    - 对 MFCC 序列进行填充，使其在批次内长度一致。
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
  
    images, mfccs, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    mfccs_padded = pad_sequence(mfccs, batch_first=True, padding_value=0.0)
  
    return images, mfccs_padded, labels

class SpecAugment(nn.Module):
    """对频谱图应用频率和时间遮蔽 (masking) 的数据增强层。"""
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

# [NEW] Mixup 辅助函数 (学习自脚本1)
def mixup_data(x_img, x_mfcc, y, alpha=0.4):
    '''返回混合后的输入、成对的目标和混合系数 lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_img.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x_img = lam * x_img + (1 - lam) * x_img[index, :]
    mixed_x_mfcc = lam * x_mfcc + (1 - lam) * x_mfcc[index, :]
    y_a, y_b = y, y[index]
    return mixed_x_img, mixed_x_mfcc, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''计算 Mixup 损失'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==============================================================================
# SECTION 2: MODEL ARCHITECTURES
# ==============================================================================

class ImageOnlyModel(nn.Module):
    """阶段一: 纯图像分类模型，用于生成基座权重。"""
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

class AudioAttention(nn.Module):
    """音频注意力机制，用于加权 GRU 的时间步输出。"""
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
    """[MODIFIED] 阶段二: 融合模型，使用 CRNN 音频分支和特征投影。"""
    def __init__(self, model_name, num_classes, n_mfcc_features, gru_hidden_size=128, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=0, drop_path_rate=drop_path_rate)
      
        num_cnn_features = self.backbone.num_features
        gru_output_features = gru_hidden_size * 2
      
        # [NEW] 1D-CNN 前端，用于从 MFCC 中提取局部特征
        cnn1d_out_channels = 128
        self.audio_cnn_frontend = nn.Sequential(
            nn.Conv1d(in_channels=n_mfcc_features, out_channels=cnn1d_out_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(cnn1d_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=cnn1d_out_channels, out_channels=cnn1d_out_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(cnn1d_out_channels),
            nn.ReLU(inplace=True)
        )
      
        self.mfcc_gru = nn.GRU(
            input_size=cnn1d_out_channels, # GRU 的输入是 CNN 的输出
            hidden_size=gru_hidden_size, 
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.audio_attention = AudioAttention(gru_output_features)
      
        # [NEW] 特征投影层，用于对齐特征空间
        projection_dim = 256 
        self.cnn_projector = nn.Linear(num_cnn_features, projection_dim)
        self.audio_projector = nn.Linear(gru_output_features, projection_dim)
      
        total_features = projection_dim * 2
      
        print(f"音频分支: MFCC({n_mfcc_features}) -> CNN1D({cnn1d_out_channels}) -> GRU({gru_output_features}) -> Projector({projection_dim})")
        print(f"图像分支: CNN({num_cnn_features}) -> Projector({projection_dim})")
      
        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(total_features),
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
      
    def forward(self, image, mfcc_sequence):
        # 图像分支
        cnn_features = self.backbone(image)
      
        # 音频分支 (CRNN)
        # 维度转换: (batch, seq_len, features) -> (batch, features, seq_len) 以适应 Conv1d
        mfcc_permuted = mfcc_sequence.permute(0, 2, 1)
        cnn_frontend_features = self.audio_cnn_frontend(mfcc_permuted)
        # 维度转换回来: (batch, features, seq_len) -> (batch, seq_len, features) 以适应 GRU
        gru_input = cnn_frontend_features.permute(0, 2, 1)
      
        gru_outputs, _ = self.mfcc_gru(gru_input)
        audio_features = self.audio_attention(gru_outputs)
      
        # 特征投影与融合
        cnn_projected = self.cnn_projector(cnn_features)
        audio_projected = self.audio_projector(audio_features)
        combined_features = torch.cat([cnn_projected, audio_projected], dim=1)
      
        output = self.classifier_head(combined_features)
        return output

# ==============================================================================
# SECTION 3: TRAINING & EVALUATION FUNCTIONS
# ==============================================================================

def run_phase1_training(train_loader, val_loader, num_classes):
    """训练 ImageOnlyModel 以生成基座权重。"""
    print("\n--- 开始阶段一: 纯图像基座模型训练 ---")
    model = ImageOnlyModel(model_name=MODEL_HUB_ID, num_classes=num_classes, pretrained=True, drop_path_rate=0.2)
    model.to(DEVICE)
  
    # 冻结较早的层
    for name, param in model.backbone.named_parameters():
        if name.startswith('stem') or name.startswith('stages.0'):
            param.requires_grad = False
          
    finetune_params = [p for name, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.classifier_head.parameters())
    param_groups = [{'params': finetune_params, 'lr': 1e-4}, {'params': head_params, 'lr': 1e-3}]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  
    best_val_loss = float('inf')
    epochs_no_improve = 0
  
    for epoch in range(PHASE1_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"阶段一 - Epoch {epoch+1}/{PHASE1_EPOCHS} [训练]", leave=False)
        for images, _, labels in pbar:
            if images.nelement() == 0: continue
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
      
        train_loss_epoch = running_loss / len(train_loader.dataset)
      
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _, labels in val_loader:
                if images.nelement() == 0: continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
      
        val_loss_epoch = val_loss / len(val_loader.dataset)
        current_lr = optimizer.param_groups[-1]['lr']
        print(f"阶段一 - Epoch {epoch+1}/{PHASE1_EPOCHS} | 训练损失: {train_loss_epoch:.4f} | 验证损失: {val_loss_epoch:.4f} | 分类头LR: {current_lr:.6f}")
      
        if scheduler: scheduler.step()
      
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), IMAGE_MODEL_WEIGHTS_PATH)
            print(f"验证损失改善至 {best_val_loss:.4f}。保存权重到 {IMAGE_MODEL_WEIGHTS_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
      
        if epochs_no_improve >= PHASE1_PATIENCE:
            print(f"\n在阶段一触发提前停止 (Early stopping)。")
            break
    print(f"\n--- 阶段一训练完成。最佳验证损失: {best_val_loss:.4f} ---")


def run_phase2_training_and_eval(train_loader, val_loader, test_loader, num_classes, display_class_labels):
    """[MODIFIED] 训练并评估最终的 CRNN 融合模型 (带 Mixup)。"""
    print("\n--- 开始阶段二: CRNN 融合模型训练 (带 Mixup) ---")
    model = AttentionFusionModel(model_name=MODEL_HUB_ID, num_classes=num_classes, n_mfcc_features=N_MFCC)
  
    print(f"\n正在从以下路径加载预训练的 backbone 权重: {IMAGE_MODEL_WEIGHTS_PATH}")
    image_only_state_dict = torch.load(IMAGE_MODEL_WEIGHTS_PATH, map_location=DEVICE)
    backbone_weights = {k: v for k, v in image_only_state_dict.items() if k.startswith('backbone.')}
    model.load_state_dict(backbone_weights, strict=False)
    print("成功将 backbone 权重加载到融合模型中。")
    model.to(DEVICE)

    print("设置优化器，采用差分学习率。")
    backbone_params = model.backbone.parameters()
    # Head 部分现在包括了新的 CNN 前端、GRU、Attention 和投影层
    head_params = list(model.audio_cnn_frontend.parameters()) + \
                  list(model.mfcc_gru.parameters()) + \
                  list(model.audio_attention.parameters()) + \
                  list(model.cnn_projector.parameters()) + \
                  list(model.audio_projector.parameters()) + \
                  list(model.classifier_head.parameters())

    optimizer = optim.AdamW([
        {'params': head_params, 'lr': 1e-3}, 
        {'params': backbone_params, 'lr': 1e-5}
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_save_path = f'best_{FINAL_MODEL_FILENAME_TAG}.pth'

    for epoch in range(PHASE2_EPOCHS):
        if epoch == 0:
            print(f"--- 步骤 A: 仅训练新增模块 (Epochs 1-{PHASE2_FREEZE_EPOCHS}) ---")
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif epoch == PHASE2_FREEZE_EPOCHS:
            print(f"\n--- 步骤 B: 解冻 Backbone 并共同训练 (Epoch {PHASE2_FREEZE_EPOCHS+1} 之后) ---")
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("Backbone 参数已解冻，将以较低学习率参与训练。")

        # 训练循环 (带 Mixup)
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"阶段二 - Epoch {epoch+1}/{PHASE2_EPOCHS} [训练]", leave=False)
        for images, mfccs, labels in pbar:
            if images.nelement() == 0: continue
            images, mfccs, labels = images.to(DEVICE), mfccs.to(DEVICE), labels.to(DEVICE)
          
            # [MODIFIED] 应用 Mixup
            mixed_images, mixed_mfccs, labels_a, labels_b, lam = mixup_data(images, mfccs, labels, alpha=0.4)
          
            optimizer.zero_grad()
            outputs = model(mixed_images, mixed_mfccs)
          
            # [MODIFIED] 使用 Mixup 损失函数
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
      
        train_loss_epoch = running_loss / len(train_loader.dataset)

        # 验证循环 (不使用 Mixup)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, mfccs, labels in val_loader:
                if images.nelement() == 0: continue
                images, mfccs, labels = images.to(DEVICE), mfccs.to(DEVICE), labels.to(DEVICE)
                outputs = model(images, mfccs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
      
        val_loss_epoch = val_loss / len(val_loader.dataset)
        head_lr = optimizer.param_groups[0]['lr']
        backbone_lr = optimizer.param_groups[1]['lr']
        print(f"阶段二 - Epoch {epoch+1}/{PHASE2_EPOCHS} | 训练损失: {train_loss_epoch:.4f} | 验证损失: {val_loss_epoch:.4f} | LRs (Head/Backbone): {head_lr:.6f}/{backbone_lr:.6f}")
      
        if scheduler: scheduler.step()
      
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"验证损失改善至 {best_val_loss:.4f}。保存模型到 {model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
      
        if epochs_no_improve >= PHASE2_PATIENCE:
            print(f"\n在阶段二触发提前停止 (Early stopping)。")
            break
          
    # --- 在测试集上进行最终评估 ---
    print(f"\n训练完成。正在从 {model_save_path} 加载最佳模型进行最终评估。")
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    model.eval()
  
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, mfccs, labels in tqdm(test_loader, desc="在测试集上评估"):
            if images.nelement() == 0: continue
            images, mfccs = images.to(DEVICE), mfccs.to(DEVICE)
            outputs = model(images, mfccs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
          
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'\n最终测试集准确率: {accuracy * 100:.2f}%')
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=display_class_labels, zero_division=0))
  
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_class_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap='Greens', values_format='d', ax=ax, xticks_rotation='vertical')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f'Confusion Matrix ({FINAL_MODEL_FILENAME_TAG})', fontsize=16, pad=20)
    fig.tight_layout()
    filename = f"Confusion_Matrix_{FINAL_MODEL_FILENAME_TAG}.png"
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\n已将混淆矩阵图保存至: {filename}")
    plt.close(fig)

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # --- 延迟启动 ---
    DELAY_HOURS = 1.5
    print(f"脚本将在 {DELAY_HOURS} 小时后启动...")
    time.sleep(DELAY_HOURS * 3600)
    print("延迟结束，脚本开始执行。")
  
    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录 '{DATA_DIR}' 不存在。请检查路径配置。")
        sys.exit(1)
      
    ORIGINAL_CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')])
    if not ORIGINAL_CLASS_NAMES:
        print(f"错误：在 '{DATA_DIR}' 中没有找到任何分类子目录。")
        sys.exit(1)
  
    NUM_CLASSES = len(ORIGINAL_CLASS_NAMES)
    print(f"找到 {NUM_CLASSES} 个分类: {ORIGINAL_CLASS_NAMES}")
  
    if NUM_CLASSES != 7:
        print(f"警告: 找到了 {NUM_CLASSES} 个类别，但期望是7个。显示的标签可能不准确。")
        DISPLAY_CLASS_NAMES = [f"Class_{i+1}" for i in range(NUM_CLASSES)]
    else:
        DISPLAY_CLASS_NAMES = [f"EC{i+1}" for i in range(NUM_CLASSES)]
    print(f"报告中将使用的显示类别名为: {DISPLAY_CLASS_NAMES}")

    # --- 数据加载逻辑: 每类随机抽样 ---
    all_paths, all_labels = [], []
    SAMPLES_PER_CLASS = 5000
    print(f"\n正在从每个类别中随机抽样最多 {SAMPLES_PER_CLASS} 个样本...")

    for i, class_name in enumerate(ORIGINAL_CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        class_image_files = [
            os.path.join(class_dir, filename)
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        random.shuffle(class_image_files)
        selected_files = class_image_files[:SAMPLES_PER_CLASS]

        if len(selected_files) < SAMPLES_PER_CLASS:
            print(f"  - 警告: 类别 '{class_name}' 只有 {len(selected_files)} 个样本, 少于期望的 {SAMPLES_PER_CLASS} 个。")
        else:
            print(f"  - 类别 '{class_name}': 使用 {len(selected_files)} 个样本。")

        all_paths.extend(selected_files)
        all_labels.extend([i] * len(selected_files))
              
    # --- 数据集划分 (70% 训练, 10% 验证, 20% 测试) ---
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, random_state=42, stratify=all_labels)
  
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.125, random_state=42, stratify=train_val_labels)

    print(f"\n--- 数据集划分摘要 ---")
    print(f"总样本数: {len(all_paths)}")
    print(f"训练集: {len(train_paths)} | 验证集: {len(val_paths)} | 测试集: {len(test_paths)}")
    print("--------------------------\n")

    # --- 数据变换 ---
    data_config = timm.data.resolve_model_data_config(MODEL_HUB_ID)
    input_size = data_config['input_size'][1]
    norm_mean, norm_std = (data_config['mean'][0],), (data_config['std'][0],)
    print(f"使用模型特定的归一化参数: mean={norm_mean}, std={norm_std}")
  
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        SpecAugment(freq_mask_param=20, time_mask_param=40, num_freq_masks=2, num_time_masks=2),
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_dataset = SpectrogramDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = SpectrogramDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = SpectrogramDataset(test_paths, test_labels, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_mfcc_and_filter_none)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_mfcc_and_filter_none)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_mfcc_and_filter_none)

    # --- 执行训练流程 ---
    if not os.path.exists(IMAGE_MODEL_WEIGHTS_PATH):
        run_phase1_training(train_loader, val_loader, NUM_CLASSES)
    else:
        print(f"\n在 '{IMAGE_MODEL_WEIGHTS_PATH}' 路径下找到已存在的基座模型权重。将跳过第一阶段训练。")

    run_phase2_training_and_eval(
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        num_classes=NUM_CLASSES, 
        display_class_labels=DISPLAY_CLASS_NAMES
    )

    print("\n--- 所有流程执行完毕！ ---")