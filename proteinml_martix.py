import os
import cv2
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.svm import SVC
from xgboost import XGBClassifier
import concurrent.futures
from sklearn.neural_network import MLPClassifier  # 新增：导入MLPClassifier
from imblearn.over_sampling import SMOTE  # 新增：导入SMOTE
from imblearn.pipeline import Pipeline  # 新增：导入Pipeline
import matplotlib.pyplot as plt  # 新增：导入matplotlib用于绘图
import matplotlib

# 设置全局字体为 SimHei
matplotlib.rcParams['font.family'] = 'SimHei'
# 解决负号 '-' 显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

train_dir = r'C:\Users\Administrator\Desktop\spectrogram\train'
test_dir = r'C:\Users\Administrator\Desktop\spectrogram\test'

def data_augmentation(img):
    augmented_images = [img]

    # 添加高斯噪声
    noise = np.random.normal(0.0001, 0.005, img.shape)
    augmented_images.append(np.clip(img + noise, 0, 1))

    # 水平翻转
    augmented_images.append(np.fliplr(img))

    # 垂直翻转
    augmented_images.append(np.flipud(img))
    
    return augmented_images

def extract_features_from_spectrogram(spectrogram):
    # 使用 librosa 提取频谱图的 MFCC 特征
    mfccs = librosa.feature.mfcc(S=spectrogram, sr=22050, n_mfcc=80)
    mfccs_mean = np.mean(mfccs, axis=1)

    # 组合所有特征
    features = np.hstack([mfccs_mean])
    return features

def process_image(img_path, label):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0  # 标准化频谱图
        augmented_images = data_augmentation(img)
        features = []
        for aug_img in augmented_images:
            img_features = extract_features_from_spectrogram(aug_img)
            features.append(img_features)
        return features, int(label)
    return None

def load_and_extract_features(folder):
    features = []
    labels = []
    print(f"加载并提取 {folder} 中的特征...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for label in os.listdir(folder):
            label_path = os.path.join(folder, label)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    img_path = os.path.join(label_path, filename)
                    futures.append(executor.submit(process_image, img_path, label))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                img_features, img_label = result
                features.extend(img_features)
                labels.extend([img_label] * len(img_features))

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # 加载训练和测试集并提取特征
    print("开始加载并提取训练和测试数据的特征...")
    X_train, y_train = load_and_extract_features(train_dir)
    X_test, y_test = load_and_extract_features(test_dir)
    
    # 检查类别分布
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"训练集中的类别: {unique_classes}")
    print(f"每个类别的样本数: {counts}")
    
    if len(unique_classes) < 2:
        raise ValueError(f"训练集中只有一个类别: {unique_classes[0]}，需要至少两个类别进行分类。")
    
    min_samples = counts.min()
    n_splits = min(5, min_samples)  # 动态调整折数，至少为每个类别的最小样本数

    print(f"使用 {n_splits} 折进行交叉验证。")
    
    # 特征选择
    selector = SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=42), threshold='median')
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # 递归特征消除（RFE）
    print("应用递归特征消除（RFE）...")
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    n_features_to_select = min(50, X_train_selected.shape[1])  # 确保不超过可用特征数量
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    X_train_rfe = rfe.fit_transform(X_train_selected, y_train)
    X_test_rfe = rfe.transform(X_test_selected)
    print(f"RFE后的训练特征维度: {X_train_rfe.shape}")
    print(f"RFE后的测试特征维度: {X_test_rfe.shape}")
    
    # 查看过采样前的样本数量
    print(f"过采样前的训练样本数量: {X_train_rfe.shape[0]}")
    
    # 应用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_rfe, y_train)
    print(f"过采样后的训练样本数量: {X_resampled.shape[0]}")
    
    # 更新训练数据为过采样后的数据
    X_train_rfe, y_train = X_resampled, y_resampled
    
    # 定义模型
    rf = RandomForestClassifier(
        n_estimators=1000,  # 树的数量
        max_depth=10,      # 树的最大深度
        min_samples_split=10,  # 最小样本分裂数
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=1000,  # 树数量
        learning_rate=0.01,  # 学习率
        max_depth=6,         # 树的最大深度
        reg_lambda=1,        # L2 正则化
        alpha=0.001,         # L1 正则化
        random_state=42,
        eval_metric='mlogloss'
    )

    svm = SVC(
        kernel='rbf',     # 核函数类型
        C=0.001,               # 正则化参数
        probability=True,    # 预测概率
        random_state=42
    )

    mlp = MLPClassifier(  # 新增：定义MLP模型
        hidden_layer_sizes=(64,32),  # 隐藏层大小
        activation='relu',           # 激活函数
        solver='adam',              # 优化算法
        max_iter=2000,               # 最大迭代次数
        random_state=42,
        alpha=0.001
    )

    # 创建投票分类器，使用调优后的模型
    print('创建投票器......')
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('svm', svm), ('mlp', mlp)],
        voting='soft',
        weights = [1, 1, 1, 1]
    )

    # 创建包含SMOTE的Pipeline
    pipeline = Pipeline([
        ('classifier', voting_clf)
    ])

    # 使用调整后的StratifiedKFold进行交叉验证
    print('开始进行交叉检验......')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(
            pipeline,
            X_train_rfe,
            y_train,
            cv=skf,
            n_jobs=-1,
            scoring='accuracy'
        )
        print(f"交叉验证分数: {cv_scores}")
        print(f"平均交叉验证分数: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    except ValueError as e:
        print(f"交叉验证过程中发生错误: {e}")
        # 这里可以选择进一步处理，例如调整n_splits或数据集
    
    # 在整个训练集上训练模型
    print('开始训练')
    pipeline.fit(X_train_rfe, y_train)

    # 预测与评估
    print('开始预测与评估')
    y_pred = pipeline.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'准确率: {accuracy * 100:.2f}%')
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    # 绘制混淆矩阵
    print('绘制混淆矩阵...')
    cm = confusion_matrix(y_test, y_pred)
    class_names = [
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # 创建更高分辨率的图形
    plt.figure(figsize=(20, 16), dpi=300)  # 增加图形大小和DPI
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Protein Classification Confusion Matrix', fontsize=16, fontfamily='Times New Roman')
    plt.xlabel('Predicted Label', fontsize=14, fontfamily='Times New Roman')
    plt.ylabel('True Label', fontsize=14, fontfamily='Times New Roman')
    
    # 调整标签显示
    plt.xticks(rotation=45, ha='right', fontfamily='Times New Roman')  # 添加Times New Roman字体
    plt.yticks(fontfamily='Times New Roman')  # 为y轴标签也设置Times New Roman字体
    plt.tight_layout()  # 自动调整布局以确保所有标签都可见
    plt.show()

    # 训练模型以进行特征重要性分析
    print('开始特征重要性分析')
    rf.fit(X_train_rfe, y_train)

    # 获取特征重要性
    feature_importances = rf.feature_importances_

    # 打印特征重要性
    print("特征重要性分析:")
    for name, model in voting_clf.named_estimators_.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\n{name} 特征重要性:")
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            print("Top 10 最重要的特征:")
            for idx in sorted_idx[-10:]:
                print(f"特征 {idx}: {feature_importance[idx]:.4f}")