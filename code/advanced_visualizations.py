# -*- coding: utf-8 -*-
"""
高级可视化图表生成脚本
为论文补充多样化、专业化的图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 先设置seaborn主题
sns.set_theme(style="whitegrid", palette="muted")

# 再设置中文字体 - 使用addfont方法（必须在sns.set_theme之后）
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('figures/advanced', exist_ok=True)

print("开始生成高级可视化图表...")

# 读取数据
try:
    features_train = pd.read_csv('data/aligned_features_train.csv')
    labels_train = pd.read_csv('附件1/训练集标签.csv')

    # 合并数据
    train_data = pd.merge(features_train, labels_train, on='样本ID')
    print(f"成功加载训练数据: {train_data.shape}")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit(1)

# ============================================================
# 图1: 相关系数热力图 (Correlation Heatmap with Clustering)
# ============================================================
print("\n生成图1: 特征相关性热力图...")

# 选择部分关键特征进行可视化（所有特征太多）
feature_cols = [col for col in train_data.columns if col not in ['样本ID', '情感标签']]
# 随机采样50个特征
np.random.seed(42)
selected_features = np.random.choice(feature_cols, size=min(50, len(feature_cols)), replace=False)

# 计算相关系数矩阵
corr_matrix = train_data[selected_features].corr()

# 绘制聚类热力图
fig, ax = plt.subplots(figsize=(12, 10))
sns.clustermap(corr_matrix,
               cmap='RdBu_r',
               center=0,
               vmin=-1, vmax=1,
               linewidths=0,
               cbar_kws={'label': '相关系数'},
               figsize=(12, 10),
               dendrogram_ratio=0.1,
               fmt='.2f')
plt.suptitle('特征相关性聚类热力图', fontsize=14, y=0.98)
plt.savefig('figures/advanced/correlation_clustermap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/correlation_clustermap.png")

# ============================================================
# 图2: 联合分布图 (Joint Distribution Plot)
# ============================================================
print("\n生成图2: 情感标签与主成分联合分布图...")

from sklearn.decomposition import PCA

# PCA降维到2维
X = train_data[feature_cols].fillna(0)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# 创建DataFrame
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    '情感标签': train_data['情感标签']
})

# 绘制联合分布图
g = sns.jointplot(data=pca_df, x='PC1', y='PC2', hue='情感标签',
                  kind='kde', fill=True, palette='coolwarm',
                  height=10, ratio=5, space=0.2)
g.fig.suptitle('情感标签在PCA空间的联合分布', fontsize=14, y=1.02)
g.set_axis_labels(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.1%})',
                  f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.1%})')
plt.savefig('figures/advanced/joint_distribution_pca.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/joint_distribution_pca.png")

# ============================================================
# 图3: 小提琴图 (Violin Plot with Swarm)
# ============================================================
print("\n生成图3: 情感标签分布的小提琴图...")

# 将情感标签分段
train_data['情感类别'] = pd.cut(train_data['情感标签'],
                              bins=[-3, -1.5, -0.5, 0.5, 1.5, 3],
                              labels=['强消极', '弱消极', '中性', '弱积极', '强积极'])

fig, ax = plt.subplots(figsize=(12, 7))
sns.violinplot(data=train_data, x='情感类别', y='情感标签',
               palette='Set2', inner='quartile', ax=ax)
sns.swarmplot(data=train_data.sample(min(200, len(train_data)), random_state=42),
              x='情感类别', y='情感标签',
              color='black', alpha=0.3, size=2, ax=ax)
ax.set_title('不同情感类别的标签分布', fontsize=14, fontweight='bold')
ax.set_xlabel('情感类别', fontsize=12)
ax.set_ylabel('情感标签值', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/advanced/emotion_violin_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/emotion_violin_plot.png")

# ============================================================
# 图4: 雷达图 (Radar Chart) - 模态特征对比
# ============================================================
print("\n生成图4: 模态特征统计雷达图...")

# 计算三个模态的统计特征
text_features = [col for col in feature_cols if 'text' in col.lower()][:10]
visual_features = [col for col in feature_cols if 'visual' in col.lower()][:10]
audio_features = [col for col in feature_cols if 'audio' in col.lower()][:10]

# 计算不同情感类别下的平均特征值
categories = ['强消极', '弱消极', '中性', '弱积极', '强积极']
text_means = []
visual_means = []
audio_means = []

for cat in categories:
    mask = train_data['情感类别'] == cat
    if len(text_features) > 0:
        text_means.append(train_data.loc[mask, text_features].mean().mean())
    else:
        text_means.append(0)
    if len(visual_features) > 0:
        visual_means.append(train_data.loc[mask, visual_features].mean().mean())
    else:
        visual_means.append(0)
    if len(audio_features) > 0:
        audio_means.append(train_data.loc[mask, audio_features].mean().mean())
    else:
        audio_means.append(0)

# 标准化
text_means = (np.array(text_means) - np.min(text_means)) / (np.max(text_means) - np.min(text_means) + 1e-10)
visual_means = (np.array(visual_means) - np.min(visual_means)) / (np.max(visual_means) - np.min(visual_means) + 1e-10)
audio_means = (np.array(audio_means) - np.min(audio_means)) / (np.max(audio_means) - np.min(audio_means) + 1e-10)

# 绘制雷达图
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
text_means = text_means.tolist()
visual_means = visual_means.tolist()
audio_means = audio_means.tolist()

# 闭合
angles += angles[:1]
text_means += text_means[:1]
visual_means += visual_means[:1]
audio_means += audio_means[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, text_means, 'o-', linewidth=2, label='文本模态', color='#1f77b4')
ax.fill(angles, text_means, alpha=0.25, color='#1f77b4')
ax.plot(angles, visual_means, 's-', linewidth=2, label='视觉模态', color='#ff7f0e')
ax.fill(angles, visual_means, alpha=0.25, color='#ff7f0e')
ax.plot(angles, audio_means, '^-', linewidth=2, label='音频模态', color='#2ca02c')
ax.fill(angles, audio_means, alpha=0.25, color='#2ca02c')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.set_title('不同情感类别下的模态特征强度', fontsize=14, pad=20, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/advanced/modality_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/modality_radar_chart.png")

# ============================================================
# 图5: 箱线图矩阵 (Boxplot Matrix)
# ============================================================
print("\n生成图5: 特征分布箱线图矩阵...")

# 选择6个代表性特征
key_features = feature_cols[:6] if len(feature_cols) >= 6 else feature_cols

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    sns.boxplot(data=train_data, x='情感类别', y=feature,
                palette='Set3', ax=axes[idx])
    axes[idx].set_title(f'{feature[:30]}...', fontsize=10)
    axes[idx].set_xlabel('')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('关键特征在不同情感类别下的分布', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/advanced/feature_boxplot_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/feature_boxplot_matrix.png")

# ============================================================
# 图6: 时序分解图 (Time Series Decomposition Simulation)
# ============================================================
print("\n生成图6: 情感标签的分布分解图...")

# 按样本顺序模拟"时序"
train_data_sorted = train_data.sort_values('情感标签').reset_index(drop=True)
y = train_data_sorted['情感标签'].values

# 模拟趋势、季节性和残差
from scipy.signal import savgol_filter

trend = savgol_filter(y, window_length=51, polyorder=3)
seasonal = y - trend
seasonal_smooth = savgol_filter(seasonal, window_length=31, polyorder=2)
residual = seasonal - seasonal_smooth

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

axes[0].plot(y, linewidth=1, color='black')
axes[0].set_title('原始情感标签分布', fontsize=12, fontweight='bold')
axes[0].set_ylabel('情感值', fontsize=10)
axes[0].grid(alpha=0.3)

axes[1].plot(trend, linewidth=2, color='blue')
axes[1].set_title('趋势成分', fontsize=12, fontweight='bold')
axes[1].set_ylabel('趋势', fontsize=10)
axes[1].grid(alpha=0.3)

axes[2].plot(seasonal_smooth, linewidth=1.5, color='green')
axes[2].set_title('周期性成分', fontsize=12, fontweight='bold')
axes[2].set_ylabel('周期波动', fontsize=10)
axes[2].grid(alpha=0.3)

axes[3].plot(residual, linewidth=0.8, color='red', alpha=0.7)
axes[3].set_title('残差成分', fontsize=12, fontweight='bold')
axes[3].set_ylabel('残差', fontsize=10)
axes[3].set_xlabel('样本索引', fontsize=10)
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/advanced/emotion_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/emotion_decomposition.png")

# ============================================================
# 图7: 3D散点图 (3D Scatter Plot)
# ============================================================
print("\n生成图7: PCA三维散点图...")

from mpl_toolkits.mplot3d import Axes3D

# PCA降维到3维
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 按情感标签着色
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                    c=train_data['情感标签'], cmap='coolwarm',
                    s=20, alpha=0.6, edgecolors='k', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=11)
ax.set_title('情感数据的三维PCA可视化', fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('情感标签', fontsize=11)

plt.tight_layout()
plt.savefig('figures/advanced/pca_3d_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/pca_3d_scatter.png")

# ============================================================
# 图8: 六边形分箱图 (Hexbin Plot)
# ============================================================
print("\n生成图8: 特征密度六边形分箱图...")

fig, ax = plt.subplots(figsize=(10, 8))
hexbin = ax.hexbin(X_pca[:, 0], X_pca[:, 1], C=train_data['情感标签'],
                   gridsize=30, cmap='RdYlGn', reduce_C_function=np.mean,
                   mincnt=1, edgecolors='black', linewidths=0.2)
ax.set_xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
ax.set_ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
ax.set_title('PCA空间的情感标签密度分布', fontsize=14, fontweight='bold')
cbar = plt.colorbar(hexbin, ax=ax)
cbar.set_label('平均情感标签', fontsize=11)
plt.tight_layout()
plt.savefig('figures/advanced/pca_hexbin_density.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: figures/advanced/pca_hexbin_density.png")

print("\n" + "="*60)
print("✓ 所有高级可视化图表生成完成!")
print("  共生成8张专业图表,保存在 figures/advanced/ 目录")
print("="*60)
