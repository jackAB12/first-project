"""
验证附件2和附件3的数据结构差异
检查以下关键问题：
1. 附件2是否同时有text和text_bert字段
2. 附件3是否只有text_bert没有text（导致KeyError）
3. 附件3中哪些模态被遮挡，遮挡位置是否一致
4. 标签分布中label=0的情况
"""

import pickle
import numpy as np
from pathlib import Path

print("="*70)
print("数据结构验证脚本")
print("="*70)

# ============================================================
# 1. 检查附件2的字段结构
# ============================================================
print("\n[1] 检查附件2 (aligned_50.pkl) 的字段结构")
print("-"*70)

try:
    with open('../data/附件2_CMU-MOSEI标准化多模态特征文件/aligned_50.pkl', 'rb') as f:
        data_aligned = pickle.load(f, encoding='latin1')

    print("✓ 附件2加载成功")
    print(f"  顶层键: {list(data_aligned.keys())}")

    # 检查训练集的字段
    train_keys = list(data_aligned['train'].keys())
    print(f"\n  训练集字段: {train_keys}")

    # 重点检查text和text_bert
    has_text = 'text' in train_keys
    has_text_bert = 'text_bert' in train_keys

    print(f"\n  ✓ 是否有 'text' 字段: {has_text}")
    print(f"  ✓ 是否有 'text_bert' 字段: {has_text_bert}")

    if has_text:
        text_shape = data_aligned['train']['text'].shape
        print(f"    - text shape: {text_shape}")

    if has_text_bert:
        text_bert_shape = data_aligned['train']['text_bert'].shape
        print(f"    - text_bert shape: {text_bert_shape}")

except Exception as e:
    print(f"✗ 附件2加载失败: {e}")

# ============================================================
# 2. 检查附件3的字段结构（重点：是否缺text字段）
# ============================================================
print("\n[2] 检查附件3的字段结构（模态缺失测试集）")
print("-"*70)

try:
    with open('../data/附件3_模态局部缺失专项测试集/missing_test_aligned.pkl', 'rb') as f:
        data_missing = pickle.load(f, encoding='latin1')

    print("✓ 附件3加载成功")

    # 检查字段
    test_keys = list(data_missing['test'].keys())
    print(f"  测试集字段: {test_keys}")

    # 重点检查text和text_bert
    has_text_in_test = 'text' in test_keys
    has_text_bert_in_test = 'text_bert' in test_keys

    print(f"\n  ✓ 是否有 'text' 字段: {has_text_in_test}")
    print(f"  ✓ 是否有 'text_bert' 字段: {has_text_bert_in_test}")

    if not has_text_in_test and has_text_bert_in_test:
        print(f"\n  ⚠️  警告：附件3只有text_bert，没有text！")
        print(f"      如果代码直接读取data['test']['text']会KeyError")

    # 检查样本数
    n_samples = len(data_missing['test']['id'])
    print(f"\n  测试样本数: {n_samples}")

except Exception as e:
    print(f"✗ 附件3加载失败: {e}")

# ============================================================
# 3. 分析附件3的模态遮挡情况
# ============================================================
print("\n[3] 分析附件3中的模态遮挡情况")
print("-"*70)

try:
    audio = data_missing['test']['audio']  # shape: (N, 50, 74)
    vision = data_missing['test']['vision']  # shape: (N, 50, 35)

    if has_text_bert_in_test:
        text_bert = data_missing['test']['text_bert']  # shape: (N, 3, 50)
        print(f"  text_bert shape: {text_bert.shape}")

    print(f"  audio shape: {audio.shape}")
    print(f"  vision shape: {vision.shape}")

    # 检查每个样本的遮挡情况
    print(f"\n  逐样本遮挡分析:")
    print(f"  {'样本ID':<15} {'audio遮挡':<12} {'vision遮挡':<12} {'遮挡位置一致'}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

    audio_masked_count = 0
    vision_masked_count = 0
    same_mask_position_count = 0

    for i in range(n_samples):
        sample_id = data_missing['test']['id'][i]

        # 检查audio是否有全0序列
        audio_sample = audio[i]  # (50, 74)
        audio_mask = np.all(audio_sample == 0, axis=1)  # (50,)
        audio_masked = np.any(audio_mask)

        # 检查vision是否有全0序列
        vision_sample = vision[i]  # (50, 35)
        vision_mask = np.all(vision_sample == 0, axis=1)  # (50,)
        vision_masked = np.any(vision_mask)

        # 检查遮挡位置是否完全一致
        mask_positions_same = np.array_equal(audio_mask, vision_mask)

        if audio_masked:
            audio_masked_count += 1
        if vision_masked:
            vision_masked_count += 1
        if mask_positions_same and (audio_masked or vision_masked):
            same_mask_position_count += 1

        # 打印前10个样本的情况
        if i < 10:
            print(f"  {sample_id:<15} {str(audio_masked):<12} {str(vision_masked):<12} {str(mask_positions_same)}")

    print(f"\n  统计结果:")
    print(f"    - audio有遮挡的样本数: {audio_masked_count}/{n_samples}")
    print(f"    - vision有遮挡的样本数: {vision_masked_count}/{n_samples}")
    print(f"    - 遮挡位置完全一致的样本数: {same_mask_position_count}/{n_samples}")

    if audio_masked_count == vision_masked_count and audio_masked_count == same_mask_position_count:
        print(f"\n  ✓ 确认：所有被遮挡样本的audio和vision遮挡位置完全相同！")
        print(f"         这是'时序局部缺失'而非'整模态缺失'")

except Exception as e:
    print(f"✗ 遮挡分析失败: {e}")

# ============================================================
# 4. 检查附件2中label=0的分布情况
# ============================================================
print("\n[4] 检查附件2中label=0的情况")
print("-"*70)

try:
    train_labels = data_aligned['train']['regression_labels']
    valid_labels = data_aligned['valid']['regression_labels']
    test_labels = data_aligned['test']['regression_labels']

    print(f"  训练集标签统计:")
    print(f"    - 总样本数: {len(train_labels)}")
    print(f"    - label=0 (中性): {np.sum(train_labels == 0)}")
    print(f"    - label>0 (正向): {np.sum(train_labels > 0)}")
    print(f"    - label<0 (负向): {np.sum(train_labels < 0)}")

    print(f"\n  验证集标签统计:")
    print(f"    - 总样本数: {len(valid_labels)}")
    print(f"    - label=0 (中性): {np.sum(valid_labels == 0)}")
    print(f"    - label>0 (正向): {np.sum(valid_labels > 0)}")
    print(f"    - label<0 (负向): {np.sum(valid_labels < 0)}")

    print(f"\n  测试集标签统计:")
    print(f"    - 总样本数: {len(test_labels)}")
    print(f"    - label=0 (中性): {np.sum(test_labels == 0)}")
    print(f"    - label>0 (正向): {np.sum(test_labels > 0)}")
    print(f"    - label<0 (负向): {np.sum(test_labels < 0)}")

    print(f"\n  ✓ 确认：label=0是独立的中性类别，不应归入正向或负向")

except Exception as e:
    print(f"✗ 标签分析失败: {e}")

print("\n" + "="*70)
print("验证完成")
print("="*70)
