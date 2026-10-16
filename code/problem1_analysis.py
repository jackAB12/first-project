"""
问题1：多模态特征提取与时序对齐
任务：分析已对齐和未对齐特征，评估对齐效果

步骤：
1. 加载附件2的已对齐和未对齐特征
2. 分析特征的维度、时序长度和对齐方式
3. 评估对齐质量
4. 可视化对齐效果
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 先设置seaborn样式
sns.set_theme(style='whitegrid', palette='Set2')

# 再设置中文字体 - 使用addfont方法（必须在sns.set_theme之后）
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path('results/problem1')
output_dir.mkdir(parents=True, exist_ok=True)

fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

print("=" * 60)
print("问题1：多模态特征提取与时序对齐")
print("=" * 60)

# ========== 1. 加载数据 ==========
print("\n[1/6] 加载附件2的特征数据...")

# 加载已对齐数据
with open('C:/Users/xieji/Desktop/E题及数据/E题数据/附件2-数据集特征文件/aligned_50.pkl', 'rb') as f:
    aligned_data = pickle.load(f)

# 加载未对齐数据
with open('C:/Users/xieji/Desktop/E题及数据/E题数据/附件2-数据集特征文件/unaligned_50.pkl', 'rb') as f:
    unaligned_data = pickle.load(f)

print(f"\n已对齐数据集划分: {list(aligned_data.keys())}")
print(f"未对齐数据集划分: {list(unaligned_data.keys())}")

# 分析已对齐数据
for split in ['train', 'valid', 'test']:
    data = aligned_data[split]
    print(f"\n=== 已对齐 {split} 数据 ===")
    print(f"样本数: {data['audio'].shape[0]}")
    print(f"文本特征(text_bert): {data['text_bert'].shape}")
    print(f"视觉特征(vision): {data['vision'].shape}")
    print(f"音频特征(audio): {data['audio'].shape}")
    print(f"回归标签(regression_labels): {data['regression_labels'].shape}")
    print(f"标签范围: [{data['regression_labels'].min():.3f}, {data['regression_labels'].max():.3f}]")

# 分析未对齐数据
print(f"\n=== 未对齐数据结构 ===")
for split in ['train', 'valid', 'test']:
    data = unaligned_data[split]
    print(f"\n{split}:")
    print(f"  样本数: {len(data['text_bert'])}")
    # 检查序列长度变化
    text_lens = [data['text_bert'][i].shape[0] for i in range(min(5, len(data['text_bert'])))]
    vision_lens = [data['vision'][i].shape[0] for i in range(min(5, len(data['vision'])))]
    audio_lens = [data['audio'][i].shape[0] for i in range(min(5, len(data['audio'])))]
    print(f"  文本序列长度示例: {text_lens}")
    print(f"  视觉序列长度示例: {vision_lens}")
    print(f"  音频序列长度示例: {audio_lens}")

# ========== 2. 统计未对齐数据的序列长度分布 ==========
print("\n[2/6] 统计未对齐数据的序列长度分布...")

def get_seq_lengths(data_split):
    """获取所有样本的序列长度"""
    text_lens = [data_split['text_bert'][i].shape[0] for i in range(len(data_split['text_bert']))]
    vision_lens = [data_split['vision'][i].shape[0] for i in range(len(data_split['vision']))]
    audio_lens = [data_split['audio'][i].shape[0] for i in range(len(data_split['audio']))]
    return np.array(text_lens), np.array(vision_lens), np.array(audio_lens)

# 训练集
train_text_lens, train_vision_lens, train_audio_lens = get_seq_lengths(unaligned_data['train'])
# 验证集
valid_text_lens, valid_vision_lens, valid_audio_lens = get_seq_lengths(unaligned_data['valid'])

print(f"\n=== 未对齐训练集序列长度统计 ===")
print(f"文本: min={train_text_lens.min()}, max={train_text_lens.max()}, mean={train_text_lens.mean():.1f}")
print(f"视觉: min={train_vision_lens.min()}, max={train_vision_lens.max()}, mean={train_vision_lens.mean():.1f}")
print(f"音频: min={train_audio_lens.min()}, max={train_audio_lens.max()}, mean={train_audio_lens.mean():.1f}")

print(f"\n=== 未对齐验证集序列长度统计 ===")
print(f"文本: min={valid_text_lens.min()}, max={valid_text_lens.max()}, mean={valid_text_lens.mean():.1f}")
print(f"视觉: min={valid_vision_lens.min()}, max={valid_vision_lens.max()}, mean={valid_vision_lens.mean():.1f}")
print(f"音频: min={valid_audio_lens.min()}, max={valid_audio_lens.max()}, mean={valid_audio_lens.mean():.1f}")

# ========== 3. 分析对齐策略 ==========
print("\n[3/6] 分析对齐策略...")

aligned_len = 50  # 已对齐后的固定长度

# 计算截断/填充比例
train_text_truncated = (train_text_lens > aligned_len).sum()
train_text_padded = (train_text_lens < aligned_len).sum()
train_text_exact = (train_text_lens == aligned_len).sum()

print(f"\n=== 文本模态对齐分析 (训练集) ===")
print(f"原始序列长度 > {aligned_len} (需要截断): {train_text_truncated} ({100*train_text_truncated/len(train_text_lens):.1f}%)")
print(f"原始序列长度 < {aligned_len} (需要填充): {train_text_padded} ({100*train_text_padded/len(train_text_lens):.1f}%)")
print(f"原始序列长度 = {aligned_len} (无需调整): {train_text_exact} ({100*train_text_exact/len(train_text_lens):.1f}%)")

# ========== 4. 特征维度统计 ==========
print("\n[4/6] 统计特征维度...")

feature_dims = {
    'train': {
        'text_bert': aligned_data['train']['text_bert'].shape[2],
        'vision': aligned_data['train']['vision'].shape[2],
        'audio': aligned_data['train']['audio'].shape[2]
    }
}

summary_data = []
for modality in ['text_bert', 'vision', 'audio']:
    summary_data.append({
        '模态': {'text_bert': '文本(BERT)', 'vision': '视觉', 'audio': '音频'}[modality],
        '特征维度': feature_dims['train'][modality],
        '对齐序列长度': aligned_len,
        '训练样本数': aligned_data['train'][modality].shape[0],
        '验证样本数': aligned_data['valid'][modality].shape[0],
        '测试样本数': aligned_data['test'][modality].shape[0]
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_dir / 'feature_summary.csv', index=False, encoding='utf-8-sig')
print(f"\n特征统计表:")
print(summary_df.to_string(index=False))
print(f"\n已保存: {output_dir / 'feature_summary.csv'}")

# ========== 5. 标签分布分析 ==========
print("\n[5/6] 分析情感标签分布...")

train_labels = aligned_data['train']['regression_labels']
valid_labels = aligned_data['valid']['regression_labels']
test_labels = aligned_data['test']['regression_labels']

label_stats = pd.DataFrame({
    '数据集': ['训练集', '验证集', '测试集'],
    '样本数': [len(train_labels), len(valid_labels), len(test_labels)],
    '最小值': [train_labels.min(), valid_labels.min(), test_labels.min()],
    '最大值': [train_labels.max(), valid_labels.max(), test_labels.max()],
    '均值': [train_labels.mean(), valid_labels.mean(), test_labels.mean()],
    '标准差': [train_labels.std(), valid_labels.std(), test_labels.std()],
    '中位数': [np.median(train_labels), np.median(valid_labels), np.median(test_labels)]
})

print(f"\n标签统计表:")
print(label_stats.to_string(index=False))
label_stats.to_csv(output_dir / 'label_statistics.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: {output_dir / 'label_statistics.csv'}")

# ========== 6. 可视化分析 ==========
print("\n[6/6] 生成可视化图表...")

# 配色方案
colors = ['#5C3D2E', '#B96420', '#F5EFE6']

# 图1: 未对齐数据的序列长度分布（多面板）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('未对齐数据的序列长度分布', fontsize=14, fontweight='bold')

modalities_data = [
    ('文本(BERT)', train_text_lens, valid_text_lens),
    ('视觉', train_vision_lens, valid_vision_lens),
    ('音频', train_audio_lens, valid_audio_lens)
]

for idx, (modality_name, train_lens, valid_lens) in enumerate(modalities_data):
    ax = axes[idx]
    ax.hist(train_lens, bins=30, alpha=0.6, label='训练集', color=colors[0], edgecolor='black')
    ax.hist(valid_lens, bins=30, alpha=0.6, label='验证集', color=colors[1], edgecolor='black')
    ax.axvline(aligned_len, color='red', linestyle='--', linewidth=2, label=f'对齐长度={aligned_len}')
    ax.set_xlabel('序列长度', fontsize=11)
    ax.set_ylabel('样本数', fontsize=11)
    ax.set_title(f'({"abc"[idx]}) {modality_name}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'problem1_unaligned_seq_length_dist.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem1_unaligned_seq_length_dist.png'}")

# 图2: 情感标签分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('情感标签分布', fontsize=14, fontweight='bold')

# (a) 直方图
ax = axes[0]
ax.hist(train_labels, bins=40, alpha=0.6, label=f'训练集 (n={len(train_labels)})',
        color=colors[0], edgecolor='black', density=True)
ax.hist(valid_labels, bins=40, alpha=0.6, label=f'验证集 (n={len(valid_labels)})',
        color=colors[1], edgecolor='black', density=True)
ax.set_xlabel('情感标签值', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.set_title('(a) 标签分布密度图', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# (b) 小提琴图
ax = axes[1]
parts = ax.violinplot([train_labels, valid_labels, test_labels],
                       positions=[1, 2, 3],
                       showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], [colors[0], colors[1], '#8B7355']):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['训练集', '验证集', '测试集'])
ax.set_ylabel('情感标签值', fontsize=11)
ax.set_title('(b) 标签分布小提琴图', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(fig_dir / 'problem1_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem1_label_distribution.png'}")

# 图3: 三模态已对齐特征示例可视化
sample_idx = 100
text_sample = aligned_data['train']['text_bert'][sample_idx].T  # (seq, feat) -> (feat, seq)
vision_sample = aligned_data['train']['vision'][sample_idx].T
audio_sample = aligned_data['train']['audio'][sample_idx].T
sample_label = aligned_data['train']['regression_labels'][sample_idx]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(f'样本#{sample_idx}的已对齐多模态特征 (标签={sample_label:.3f})',
             fontsize=14, fontweight='bold')

# (a) 文本特征
im1 = axes[0].imshow(text_sample, aspect='auto', cmap='viridis', interpolation='nearest')
axes[0].set_xlabel('时间步', fontsize=11)
axes[0].set_ylabel('BERT特征维度', fontsize=11)
axes[0].set_title(f'(a) 文本特征 (维度={text_sample.shape[0]})', fontsize=12)
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('特征值', fontsize=10)

# (b) 视觉特征
im2 = axes[1].imshow(vision_sample, aspect='auto', cmap='viridis', interpolation='nearest')
axes[1].set_xlabel('时间步', fontsize=11)
axes[1].set_ylabel('视觉特征维度', fontsize=11)
axes[1].set_title(f'(b) 视觉特征 (维度={vision_sample.shape[0]})', fontsize=12)
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label('特征值', fontsize=10)

# (c) 音频特征
im3 = axes[2].imshow(audio_sample, aspect='auto', cmap='viridis', interpolation='nearest')
axes[2].set_xlabel('时间步', fontsize=11)
axes[2].set_ylabel('音频特征维度', fontsize=11)
axes[2].set_title(f'(c) 音频特征 (维度={audio_sample.shape[0]})', fontsize=12)
cbar3 = plt.colorbar(im3, ax=axes[2])
cbar3.set_label('特征值', fontsize=10)

plt.tight_layout()
plt.savefig(fig_dir / 'problem1_aligned_feature_example.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem1_aligned_feature_example.png'}")

# 图4: 对齐效果统计
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['文本', '视觉', '音频']
truncated = [
    (train_text_lens > aligned_len).sum(),
    (train_vision_lens > aligned_len).sum(),
    (train_audio_lens > aligned_len).sum()
]
padded = [
    (train_text_lens < aligned_len).sum(),
    (train_vision_lens < aligned_len).sum(),
    (train_audio_lens < aligned_len).sum()
]
exact = [
    (train_text_lens == aligned_len).sum(),
    (train_vision_lens == aligned_len).sum(),
    (train_audio_lens == aligned_len).sum()
]

x = np.arange(len(categories))
width = 0.25

bars1 = ax.bar(x - width, truncated, width, label='截断', color='#B96420')
bars2 = ax.bar(x, padded, width, label='填充', color='#5C3D2E')
bars3 = ax.bar(x + width, exact, width, label='无需调整', color='#8B7355')

ax.set_xlabel('模态', fontsize=12)
ax.set_ylabel('样本数', fontsize=12)
ax.set_title('训练集各模态的序列对齐操作统计', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(fig_dir / 'problem1_alignment_operations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem1_alignment_operations.png'}")

print("\n" + "=" * 60)
print("问题1分析完成！")
print("=" * 60)
print(f"\n结果文件保存在: {output_dir}")
print(f"图表文件保存在: {fig_dir}")
