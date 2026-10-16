# -*- coding: utf-8 -*-
"""
补充专业可视化图表 - 第二批
包括Sankey图、网络图、热力日历等高级图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 先设置seaborn主题（如果使用）
# sns.set_theme() 会重置rcParams，所以字体设置必须在之后

# 设置中文字体 - 使用addfont方法
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import os
os.makedirs('figures/advanced', exist_ok=True)

print("开始生成第二批高级可视化图表...")

# 读取数据
try:
    features_train = pd.read_csv('data/aligned_features_train.csv')
    labels_train = pd.read_csv('附件1/训练集标签.csv')
    train_data = pd.merge(features_train, labels_train, on='样本ID')
    print(f"成功加载数据: {train_data.shape}")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit(1)

# 情感分类
train_data['情感类别'] = pd.cut(train_data['情感标签'],
                              bins=[-3, -1.5, -0.5, 0.5, 1.5, 3],
                              labels=['强消极', '弱消极', '中性', '弱积极', '强积极'])

# ============================================================
# 图9: 山脊图 (Ridgeline Plot)
# ============================================================
print("\n生成图9: 情感标签分布的山脊图...")

try:
    import joypy

    fig, axes = joypy.joyplot(train_data, column='情感标签', by='情感类别',
                              figsize=(12, 8), alpha=0.6,
                              color=['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c'])
    plt.xlabel('情感标签值', fontsize=12)
    plt.title('不同情感类别的标签分布山脊图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures/advanced/emotion_ridgeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 保存: figures/advanced/emotion_ridgeline.png")
except ImportError:
    print("  ⚠ joypy未安装,跳过山脊图")
except Exception as e:
    print(f"  ⚠ 山脊图生成失败: {e}")

# ============================================================
# 图10: 成对关系图 (Pair Plot) - 精简版
# ============================================================
print("\n生成图10: 主成分成对关系图...")

from sklearn.decomposition import PCA

feature_cols = [col for col in train_data.columns if col not in ['样本ID', '情感标签', '情感类别']]
X = train_data[feature_cols].fillna(0)

# PCA降维到4维
pca = PCA(n_components=4, random_state=42)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'PC3': X_pca[:, 2],
    'PC4': X_pca[:, 3],
    '情感类别': train_data['情感类别']
})

# 采样以加快绘制
sample_size = min(500, len(pca_df))
pca_sample = pca_df.sample(sample_size, random_state=42)

g = sns.pairplot(pca_sample, hue='情感类别', palette='Set2',
                diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20},
                corner=True, height=2.5)
g.fig.suptitle('主成分成对关系图', fontsize=14, y=1.0, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/advanced/pca_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/pca_pairplot.png")

# ============================================================
# 图11: 堆叠面积图 (Stacked Area Chart)
# ============================================================
print("\n生成图11: 情感类别分布堆叠面积图...")

# 按样本顺序统计情感类别
window_size = 100
emotion_counts = []
indices = []

for i in range(0, len(train_data) - window_size, window_size // 2):
    window = train_data.iloc[i:i+window_size]
    counts = window['情感类别'].value_counts()
    emotion_counts.append(counts)
    indices.append(i + window_size // 2)

# 构建DataFrame
emotion_df = pd.DataFrame(emotion_counts, index=indices).fillna(0)
emotion_df = emotion_df.reindex(['强消极', '弱消极', '中性', '弱积极', '强积极'], axis=1).fillna(0)

fig, ax = plt.subplots(figsize=(14, 7))
ax.stackplot(emotion_df.index, emotion_df.T.values,
            labels=emotion_df.columns,
            colors=['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c'],
            alpha=0.8)
ax.set_xlabel('样本索引', fontsize=12)
ax.set_ylabel('情感类别样本数', fontsize=12)
ax.set_title('滑动窗口内情感类别分布的堆叠面积图', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/advanced/emotion_stacked_area.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/emotion_stacked_area.png")

# ============================================================
# 图12: Q-Q图 (Quantile-Quantile Plot)
# ============================================================
print("\n生成图12: 情感标签的Q-Q图...")

from scipy import stats

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 全体样本的Q-Q图
stats.probplot(train_data['情感标签'], dist="norm", plot=axes[0])
axes[0].set_title('情感标签的正态性检验(Q-Q图)', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# 残差的Q-Q图（假设用均值预测）
residuals = train_data['情感标签'] - train_data['情感标签'].mean()
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('残差的正态性检验(Q-Q图)', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/advanced/qq_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/qq_plots.png")

# ============================================================
# 图13: ECDF图 (Empirical Cumulative Distribution Function)
# ============================================================
print("\n生成图13: 经验累积分布函数图...")

fig, ax = plt.subplots(figsize=(12, 7))

for cat in ['强消极', '弱消极', '中性', '弱积极', '强积极']:
    data = train_data[train_data['情感类别'] == cat]['情感标签'].sort_values()
    ecdf = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, ecdf, linewidth=2, label=cat, alpha=0.8)

ax.set_xlabel('情感标签值', fontsize=12)
ax.set_ylabel('累积概率', fontsize=12)
ax.set_title('不同情感类别的经验累积分布函数', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/advanced/ecdf_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/ecdf_plot.png")

# ============================================================
# 图14: 等高线密度图 (Contour Density Plot)
# ============================================================
print("\n生成图14: PCA空间的等高线密度图...")

from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(12, 9))

# 为每个情感类别绘制等高线
colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c']
categories = ['强消极', '弱消极', '中性', '弱积极', '强积极']

for cat, color in zip(categories, colors):
    data = pca_df[pca_df['情感类别'] == cat][['PC1', 'PC2']]
    if len(data) > 10:
        x = data['PC1'].values
        y = data['PC2'].values

        # 核密度估计
        try:
            kde = gaussian_kde([x, y])

            # 创建网格
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                np.linspace(y_min, y_max, 50))

            # 计算密度
            z = kde(np.vstack([xx.ravel(), yy.ravel()]))
            z = z.reshape(xx.shape)

            # 绘制等高线
            ax.contour(xx, yy, z, levels=3, colors=color, alpha=0.6, linewidths=2)
            ax.contourf(xx, yy, z, levels=3, colors=color, alpha=0.15)
        except:
            pass

# 添加散点
for cat, color in zip(categories, colors):
    data = pca_df[pca_df['情感类别'] == cat][['PC1', 'PC2']]
    sample = data.sample(min(50, len(data)), random_state=42)
    ax.scatter(sample['PC1'], sample['PC2'], c=color, s=30,
              alpha=0.6, edgecolors='black', linewidth=0.5, label=cat)

ax.set_xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
ax.set_ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
ax.set_title('PCA空间的情感类别密度等高线图', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best', framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/advanced/density_contour.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/density_contour.png")

# ============================================================
# 图15: 误差带图 (Error Band Plot)
# ============================================================
print("\n生成图15: 滑动窗口均值与置信区间图...")

window_size = 50
means = []
stds = []
indices = []

for i in range(0, len(train_data) - window_size, 10):
    window = train_data.iloc[i:i+window_size]['情感标签']
    means.append(window.mean())
    stds.append(window.std())
    indices.append(i + window_size // 2)

means = np.array(means)
stds = np.array(stds)
indices = np.array(indices)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(indices, means, linewidth=2, color='blue', label='滑动均值')
ax.fill_between(indices, means - 1.96*stds, means + 1.96*stds,
                alpha=0.3, color='blue', label='95%置信区间')
ax.fill_between(indices, means - stds, means + stds,
                alpha=0.5, color='blue', label='±1标准差')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('样本索引', fontsize=12)
ax.set_ylabel('情感标签值', fontsize=12)
ax.set_title('情感标签的滑动窗口均值与置信区间', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/advanced/error_band_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/error_band_plot.png")

# ============================================================
# 图16: 热力矩阵 (Heatmap Matrix) - 分段统计
# ============================================================
print("\n生成图16: 情感标签分段统计热力矩阵...")

# 将样本分段
n_segments = 20
train_data['段号'] = pd.cut(range(len(train_data)), bins=n_segments, labels=False)

# 统计每段每个情感类别的数量
heatmap_data = train_data.groupby(['段号', '情感类别']).size().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(['强消极', '弱消极', '中性', '弱积极', '强积极'], axis=1).fillna(0)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_data.T, cmap='YlOrRd', annot=False, fmt='d',
           cbar_kws={'label': '样本数量'}, linewidths=0.5, ax=ax)
ax.set_xlabel('数据段序号', fontsize=12)
ax.set_ylabel('情感类别', fontsize=12)
ax.set_title('情感类别在不同数据段的分布热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/advanced/segment_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/segment_heatmap.png")

print("\n" + "="*60)
print("✓ 第二批高级可视化图表生成完成!")
print("  新增8张专业图表")
print("="*60)
