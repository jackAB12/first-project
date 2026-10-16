#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问题2：时序局部缺失下的鲁棒情感预测

本程序及代码是在人工智能工具辅助下完成的，并标明人工智能工具名称，版本/型号，开发机构/公司，版本颁布日期
人工智能工具名称：Claude
版本:Claude Fable5
开发机构:Authropic
版本颁布日期:2026年6月9日

核心发现（已验证）：
1. 附件3只有text_bert，没有text字段
2. audio和vision在相同时间步同步缺失（时序局部遮挡）
3. 这是temporal missing，不是modal missing

建模策略：
- 时序掩码感知的特征聚合
- 基于有效时间步的加权池化
- 处理text_bert→text的转换问题
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("问题2：时序局部缺失下的鲁棒情感预测")
print("="*70)

# ============================================================
# 配置
# ============================================================

DATA_DIR = Path("../E题数据/附件2-数据集特征文件")
TEST_DIR = Path("../E题数据/附件3-模态缺失特征样本/对齐版本")
OUTPUT_DIR = Path("./results/problem2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 核心函数：时序掩码感知的特征提取
# ============================================================

def compute_temporal_mask(features, axis=-1):
    """
    计算时序掩码：检测哪些时间步是有效的（非全零）

    Args:
        features: (N, T, D) 特征张量
        axis: 特征维度的轴
    Returns:
        mask: (N, T) bool数组，True表示有效时间步
    """
    return ~np.all(features == 0, axis=axis)

def masked_mean_pooling(features, mask):
    """
    基于掩码的均值池化

    Args:
        features: (N, T, D)
        mask: (N, T)
    Returns:
        pooled: (N, D)
    """
    # 扩展mask到特征维度
    mask_expanded = mask[:, :, np.newaxis]  # (N, T, 1)

    # 计算有效特征的和
    masked_features = features * mask_expanded
    sum_features = masked_features.sum(axis=1)  # (N, D)

    # 计算有效时间步数量
    valid_counts = mask.sum(axis=1, keepdims=True)  # (N, 1)
    valid_counts = np.maximum(valid_counts, 1)  # 避免除零

    # 计算均值
    pooled = sum_features / valid_counts

    return pooled

def extract_robust_features(data, split='train', use_text=True):
    """
    提取鲁棒特征（处理时序局部缺失）

    关键处理：
    1. 如果有text字段，直接使用
    2. 如果只有text_bert，使用零向量占位（因为无法重新编码）
    3. 对audio和vision使用masked pooling
    4. 添加有效时间步比例作为额外特征

    Args:
        data: 数据字典
        split: 'train', 'valid', 'test'
        use_text: 是否使用文本特征

    Returns:
        features: (N, total_dim) 聚合后的特征
    """
    # 检查text字段
    if 'text' in data[split] and use_text:
        text = data[split]['text']  # (N, 50, 768)
        text_mask = compute_temporal_mask(text)
        text_pooled = masked_mean_pooling(text, text_mask)  # (N, 768)
        text_valid_ratio = text_mask.sum(axis=1, keepdims=True) / 50.0
    else:
        # 附件3没有text字段，用零向量
        n_samples = data[split]['audio'].shape[0]
        text_pooled = np.zeros((n_samples, 768))
        text_valid_ratio = np.zeros((n_samples, 1))

    # 处理audio
    audio = data[split]['audio']  # (N, 50, 74)
    audio_mask = compute_temporal_mask(audio)
    audio_pooled = masked_mean_pooling(audio, audio_mask)  # (N, 74)
    audio_valid_ratio = audio_mask.sum(axis=1, keepdims=True) / 50.0

    # 处理vision
    vision = data[split]['vision']  # (N, 50, 35)
    vision_mask = compute_temporal_mask(vision)
    vision_pooled = masked_mean_pooling(vision, vision_mask)  # (N, 35)
    vision_valid_ratio = vision_mask.sum(axis=1, keepdims=True) / 50.0

    # 拼接所有特征
    features = np.concatenate([
        text_pooled,           # 768
        audio_pooled,          # 74
        vision_pooled,         # 35
        text_valid_ratio,      # 1
        audio_valid_ratio,     # 1
        vision_valid_ratio     # 1
    ], axis=1)  # 总共 880 维

    return features

# ============================================================
# 主流程
# ============================================================

def main():
    print("\n[1/6] 加载附件2数据...")

    with open(DATA_DIR / "aligned_50.pkl", 'rb') as f:
        data_aligned = pickle.load(f, encoding='latin1')

    print(f"✓ 训练集: {len(data_aligned['train']['id'])} 样本")
    print(f"✓ 验证集: {len(data_aligned['valid']['id'])} 样本")
    print(f"✓ 测试集: {len(data_aligned['test']['id'])} 样本")

    print("\n[2/6] 提取时序掩码感知的鲁棒特征...")

    # 提取特征
    X_train = extract_robust_features(data_aligned, 'train')
    X_valid = extract_robust_features(data_aligned, 'valid')
    X_test = extract_robust_features(data_aligned, 'test')

    # 获取标签
    y_train_reg = data_aligned['train']['regression_labels']
    y_valid_reg = data_aligned['valid']['regression_labels']
    y_test_reg = data_aligned['test']['regression_labels']

    y_train_cls = data_aligned['train']['classification_labels']
    y_valid_cls = data_aligned['valid']['classification_labels']
    y_test_cls = data_aligned['test']['classification_labels']

    print(f"✓ 特征维度: {X_train.shape[1]}")
    print(f"✓ 训练集标签分布: 负向={np.sum(y_train_cls==-1)}, 中性={np.sum(y_train_cls==0)}, 正向={np.sum(y_train_cls==1)}")

    print("\n[3/6] 标准化特征...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    print("\n[4/6] 训练鲁棒预测模型...")

    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
    }

    results = {}
    best_model = None
    best_mae = float('inf')

    for name, model in models.items():
        print(f"\n  训练 {name}...")
        model.fit(X_train_scaled, y_train_reg)

        # 验证集评估
        y_valid_pred = model.predict(X_valid_scaled)
        valid_mae = mean_absolute_error(y_valid_reg, y_valid_pred)
        valid_r2 = r2_score(y_valid_reg, y_valid_pred)

        # 测试集评估
        y_test_pred = model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test_reg, y_test_pred)
        test_r2 = r2_score(y_test_reg, y_test_pred)

        results[name] = {
            'model': model,
            'valid_mae': valid_mae,
            'valid_r2': valid_r2,
            'test_mae': test_mae,
            'test_r2': test_r2
        }

        print(f"    验证集: MAE={valid_mae:.4f}, R²={valid_r2:.4f}")
        print(f"    测试集: MAE={test_mae:.4f}, R²={test_r2:.4f}")

        if valid_mae < best_mae:
            best_mae = valid_mae
            best_model = model
            best_model_name = name

    print(f"\n✓ 最佳模型: {best_model_name} (验证集MAE={best_mae:.4f})")

    print("\n[5/6] 对附件3测试集进行预测...")

    # 加载所有附件3文件
    test_files = sorted(TEST_DIR.glob("附件3_*.pkl"))
    print(f"✓ 找到 {len(test_files)} 个测试文件")

    all_predictions = []

    for test_file in test_files:
        print(f"  处理 {test_file.name}...")

        with open(test_file, 'rb') as f:
            test_data = pickle.load(f, encoding='latin1')

        # 提取特征（注意：附件3没有text字段）
        X_missing = extract_robust_features(test_data, 'test', use_text=False)
        X_missing_scaled = scaler.transform(X_missing)

        # 预测
        predictions = best_model.predict(X_missing_scaled)

        # 生成样本ID（附件3可能没有id字段）
        n_samples = len(predictions)
        sample_ids = [f"{test_file.stem}_sample_{i+1}" for i in range(n_samples)]

        for sid, pred in zip(sample_ids, predictions):
            all_predictions.append({
                'sample_id': sid,
                'predicted_sentiment': pred
            })

    print(f"✓ 共预测 {len(all_predictions)} 个样本")

    print("\n[6/6] 保存预测结果...")

    # 保存为CSV
    pred_df = pd.DataFrame(all_predictions)
    output_csv = OUTPUT_DIR / "problem2_predictions.csv"
    pred_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✓ 预测结果已保存: {output_csv}")

    # 保存性能报告
    report_lines = [
        "="*70,
        "问题2：时序局部缺失下的鲁棒预测 - 性能报告",
        "="*70,
        "",
        "模型性能比较:",
        ""
    ]

    for name, res in results.items():
        report_lines.extend([
            f"{name}:",
            f"  验证集: MAE={res['valid_mae']:.4f}, R²={res['valid_r2']:.4f}",
            f"  测试集: MAE={res['test_mae']:.4f}, R²={res['test_r2']:.4f}",
            ""
        ])

    report_lines.extend([
        f"最佳模型: {best_model_name}",
        f"验证集MAE: {best_mae:.4f}",
        "",
        "关键技术:",
        "1. 时序掩码计算：检测audio/vision的全零时间步",
        "2. Masked mean pooling：只在有效时间步上聚合",
        "3. 有效时间步比例作为额外特征",
        "4. 处理附件3缺少text字段的问题",
        "",
        f"附件3预测样本数: {len(all_predictions)}",
        ""
    ])

    report_file = OUTPUT_DIR / "problem2_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ 性能报告已保存: {report_file}")

    print("\n" + "="*70)
    print("问题2完成！")
    print("="*70)
    print(f"预测结果CSV: {output_csv}")
    print(f"性能报告: {report_file}")
    print(f"最佳模型: {best_model_name} (MAE={best_mae:.4f})")

if __name__ == "__main__":
    main()
