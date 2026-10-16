#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证附件2和附件3的数据结构差异"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pickle
import numpy as np

print("="*70)
print("数据结构验证")
print("="*70)

# 检查附件2
print("\n[1] 附件2 (aligned_50.pkl)")
with open('../E题数据/附件2-数据集特征文件/aligned_50.pkl', 'rb') as f:
    data2 = pickle.load(f, encoding='latin1')

print(f"训练集字段: {list(data2['train'].keys())}")
print(f"- 有'text'字段: {'text' in data2['train']}")
print(f"- 有'text_bert'字段: {'text_bert' in data2['train']}")

if 'text' in data2['train']:
    print(f"  text shape: {data2['train']['text'].shape}")
if 'text_bert' in data2['train']:
    print(f"  text_bert shape: {data2['train']['text_bert'].shape}")

# 检查附件3（只检查第一个文件）
print("\n[2] 附件3 (附件3_01.pkl)")
with open('../E题数据/附件3-模态缺失特征样本/对齐版本/附件3_01.pkl', 'rb') as f:
    data3 = pickle.load(f, encoding='latin1')

print(f"测试集字段: {list(data3['test'].keys())}")
print(f"- 有'text'字段: {'text' in data3['test']}")
print(f"- 有'text_bert'字段: {'text_bert' in data3['test']}")

if 'text' not in data3['test']:
    print("\n!!! 关键发现：附件3缺少'text'字段，只有'text_bert'")
    print("    如果代码用data['test']['text']会KeyError")

# 检查模态遮挡
print("\n[3] 模态遮挡分析")
# 附件3可能没有'id'字段，用audio的第一维长度获取样本数
n_samples = data3['test']['audio'].shape[0]
print(f"样本数: {n_samples}")
print(f"附件3的字段: {list(data3['test'].keys())}")

audio = data3['test']['audio']
vision = data3['test']['vision']

audio_masked = 0
vision_masked = 0
same_position = 0

for i in range(n_samples):
    # 检查是否有全0时间步
    a_mask = np.all(audio[i] == 0, axis=1)
    v_mask = np.all(vision[i] == 0, axis=1)

    a_has = np.any(a_mask)
    v_has = np.any(v_mask)

    if a_has:
        audio_masked += 1
    if v_has:
        vision_masked += 1
    if np.array_equal(a_mask, v_mask):
        same_position += 1

print(f"audio被遮挡样本数: {audio_masked}/{n_samples}")
print(f"vision被遮挡样本数: {vision_masked}/{n_samples}")
print(f"遮挡位置完全一致: {same_position}/{n_samples}")

if audio_masked == vision_masked == same_position:
    print("\n!!! 关键发现：所有被遮挡样本的audio和vision位置完全相同")
    print("    这是'时序局部缺失'，不是'整模态缺失'")

# 检查标签分布
print("\n[4] 附件2标签分布")
train_labels = data2['train']['regression_labels']
print(f"训练集: n={len(train_labels)}")
print(f"  label=0 (中性): {np.sum(train_labels == 0)}")
print(f"  label>0 (正向): {np.sum(train_labels > 0)}")
print(f"  label<0 (负向): {np.sum(train_labels < 0)}")

print("\n" + "="*70)
