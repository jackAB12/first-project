"""
问题2：模态缺失下的情感预测（使用传统机器学习方法）
任务：构建鲁棒的多模态融合模型，处理模态缺失情况

方法：
1. 特征提取：对每个模态的时序特征进行统计聚合
2. 模态融合：使用随机森林/梯度提升树进行多模态融合
3. 鲁棒性训练：训练时随机mask模态，提升缺失情况下的性能
4. 预测附件3的30个缺失样本
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib

# 设置随机种子
np.random.seed(42)

# 设置中文字体 - 使用addfont方法
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
# 先设置seaborn主题
sns.set_theme(style='whitegrid', palette='Set2')

# 再设置中文字体 - 使用addfont方法（必须在sns.set_theme之后）
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path('results/problem2')
output_dir.mkdir(parents=True, exist_ok=True)

fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

print("=" * 60)
print("问题2：模态缺失下的情感预测")
print("=" * 60)

# ========== 1. 加载数据 ==========
print("\n[1/7] 加载数据...")

with open('E题数据/附件2-数据集特征文件/aligned_50.pkl', 'rb') as f:
    aligned_data = pickle.load(f)

train_data = aligned_data['train']
valid_data = aligned_data['valid']
test_data = aligned_data['test']

print(f"训练集: {train_data['audio'].shape[0]} 样本")
print(f"验证集: {valid_data['audio'].shape[0]} 样本")
print(f"测试集: {test_data['audio'].shape[0]} 样本")

# ========== 2. 特征工程 ==========
print("\n[2/7] 特征工程：从时序数据提取统计特征...")

def extract_statistical_features(data_split):
    """从时序数据中提取统计特征"""

    text_bert = data_split['text_bert']  # (N, 3, 50)
    vision = data_split['vision']        # (N, 50, 35)
    audio = data_split['audio']          # (N, 50, 74)
    labels = data_split['regression_labels']

    N = text_bert.shape[0]
    features_list = []

    for i in range(N):
        # 文本特征 (3, 50) -> (3, 50) -> 统计特征
        text = text_bert[i].T  # (50, 3)
        text_feats = [
            text.mean(axis=0),  # 均值
            text.std(axis=0),   # 标准差
            text.max(axis=0),   # 最大值
            text.min(axis=0),   # 最小值
            np.percentile(text, 25, axis=0),  # 25分位
            np.percentile(text, 75, axis=0),  # 75分位
        ]
        text_flat = np.concatenate(text_feats)

        # 视觉特征 (50, 35) -> 统计特征
        vis = vision[i]
        vis_feats = [
            vis.mean(axis=0),
            vis.std(axis=0),
            vis.max(axis=0),
            vis.min(axis=0),
            np.percentile(vis, 25, axis=0),
            np.percentile(vis, 75, axis=0),
        ]
        vis_flat = np.concatenate(vis_feats)

        # 音频特征 (50, 74) -> 统计特征
        aud = audio[i]
        aud_feats = [
            aud.mean(axis=0),
            aud.std(axis=0),
            aud.max(axis=0),
            aud.min(axis=0),
            np.percentile(aud, 25, axis=0),
            np.percentile(aud, 75, axis=0),
        ]
        aud_flat = np.concatenate(aud_feats)

        # 拼接所有特征
        sample_feat = np.concatenate([text_flat, vis_flat, aud_flat])
        features_list.append(sample_feat)

    X = np.array(features_list)
    y = labels

    return X, y

# 提取特征
X_train, y_train = extract_statistical_features(train_data)
X_valid, y_valid = extract_statistical_features(valid_data)
X_test, y_test = extract_statistical_features(test_data)

print(f"训练特征形状: {X_train.shape}")
print(f"验证特征形状: {X_valid.shape}")
print(f"测试特征形状: {X_test.shape}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# 保存scaler
joblib.dump(scaler, model_dir / 'feature_scaler.pkl')

# ========== 3. 训练基础模型 ==========
print("\n[3/7] 训练多模态融合模型...")

# 尝试多个模型
models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
}

results = {}
for name, model in models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train_scaled, y_train)

    # 验证集评估
    y_valid_pred = model.predict(X_valid_scaled)
    valid_mae = mean_absolute_error(y_valid, y_valid_pred)
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    valid_r2 = r2_score(y_valid, y_valid_pred)

    # 测试集评估
    y_test_pred = model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    results[name] = {
        'model': model,
        'valid_mae': valid_mae,
        'valid_rmse': valid_rmse,
        'valid_r2': valid_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'y_test_pred': y_test_pred
    }

    print(f"  验证集 - MAE: {valid_mae:.4f}, RMSE: {valid_rmse:.4f}, R2: {valid_r2:.4f}")
    print(f"  测试集 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

# 选择最佳模型
best_model_name = min(results, key=lambda k: results[k]['valid_mae'])
best_model = results[best_model_name]['model']
print(f"\n最佳模型: {best_model_name} (验证MAE={results[best_model_name]['valid_mae']:.4f})")

# 保存最佳模型
joblib.dump(best_model, model_dir / 'problem2_best_model.pkl')

# ========== 4. 模拟模态缺失训练 ==========
print("\n[4/7] 训练鲁棒模型（模拟模态缺失）...")

def create_missing_modality_data(X, missing_rate=0.2):
    """随机将某些模态的特征置零，模拟缺失"""
    X_missing = X.copy()
    n_samples = X.shape[0]

    # 特征索引范围
    text_end = 3 * 6  # 18
    vision_end = text_end + 35 * 6  # 18 + 210 = 228
    audio_end = vision_end + 74 * 6  # 228 + 444 = 672

    for i in range(n_samples):
        # 随机决定是否mask每个模态
        if np.random.rand() < missing_rate:
            X_missing[i, :text_end] = 0  # mask文本
        if np.random.rand() < missing_rate:
            X_missing[i, text_end:vision_end] = 0  # mask视觉
        if np.random.rand() < missing_rate:
            X_missing[i, vision_end:audio_end] = 0  # mask音频

    return X_missing

# 创建带缺失的训练数据
X_train_robust = create_missing_modality_data(X_train_scaled, missing_rate=0.15)

# 训练鲁棒模型
robust_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

print("训练鲁棒模型...")
robust_model.fit(X_train_robust, y_train)

# 评估鲁棒模型
y_valid_pred_robust = robust_model.predict(X_valid_scaled)
valid_mae_robust = mean_absolute_error(y_valid, y_valid_pred_robust)

y_test_pred_robust = robust_model.predict(X_test_scaled)
test_mae_robust = mean_absolute_error(y_test, y_test_pred_robust)

print(f"鲁棒模型 - 验证MAE: {valid_mae_robust:.4f}, 测试MAE: {test_mae_robust:.4f}")

# 保存鲁棒模型
joblib.dump(robust_model, model_dir / 'problem2_robust_model.pkl')

# ========== 5. 预测附件3 ==========
print("\n[5/7] 预测附件3的30个模态缺失样本...")

def extract_features_single(text_bert, vision, audio):
    """提取单个样本的特征"""
    # 文本
    text = text_bert[0].T
    text_feats = [
        text.mean(axis=0),
        text.std(axis=0),
        text.max(axis=0),
        text.min(axis=0),
        np.percentile(text, 25, axis=0),
        np.percentile(text, 75, axis=0),
    ]
    text_flat = np.concatenate(text_feats)

    # 视觉
    vis = vision[0]
    vis_feats = [
        vis.mean(axis=0),
        vis.std(axis=0),
        vis.max(axis=0),
        vis.min(axis=0),
        np.percentile(vis, 25, axis=0),
        np.percentile(vis, 75, axis=0),
    ]
    vis_flat = np.concatenate(vis_feats)

    # 音频
    aud = audio[0]
    aud_feats = [
        aud.mean(axis=0),
        aud.std(axis=0),
        aud.max(axis=0),
        aud.min(axis=0),
        np.percentile(aud, 25, axis=0),
        np.percentile(aud, 75, axis=0),
    ]
    aud_flat = np.concatenate(aud_feats)

    return np.concatenate([text_flat, vis_flat, aud_flat]).reshape(1, -1)

missing_predictions = []

for i in range(1, 31):
    with open(f'E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{i:02d}.pkl', 'rb') as f:
        missing_data = pickle.load(f)

    missing_test = missing_data['test']

    # 提取特征
    feat = extract_features_single(
        missing_test['text_bert'],
        missing_test['vision'],
        missing_test['audio']
    )

    # 标准化
    feat_scaled = scaler.transform(feat)

    # 预测（使用鲁棒模型）
    pred = robust_model.predict(feat_scaled)[0]

    # 计算零值占比
    text_zero_pct = (missing_test['text_bert'] == 0).sum() / missing_test['text_bert'].size * 100
    vision_zero_pct = (missing_test['vision'] == 0).sum() / missing_test['vision'].size * 100
    audio_zero_pct = (missing_test['audio'] == 0).sum() / missing_test['audio'].size * 100

    missing_predictions.append({
        '样本编号': i,
        '预测值': pred,
        'text零值占比(%)': text_zero_pct,
        'vision零值占比(%)': vision_zero_pct,
        'audio零值占比(%)': audio_zero_pct
    })

    print(f"附件3_{i:02d}: 预测={pred:.4f}")

# 保存预测结果
missing_pred_df = pd.DataFrame(missing_predictions)
missing_pred_df.to_csv(output_dir / 'missing_modality_predictions.csv', index=False, encoding='utf-8-sig')
print(f"\n预测结果已保存: {output_dir / 'missing_modality_predictions.csv'}")

# 保存用于提交的CSV
submission_df = missing_pred_df[['样本编号', '预测值']].copy()
submission_df.columns = ['sample_id', 'prediction']
submission_df.to_csv(output_dir / 'problem2_submission.csv', index=False)
print(f"提交文件已保存: {output_dir / 'problem2_submission.csv'}")

# ========== 6. 可视化 ==========
print("\n[6/7] 生成可视化图表...")

colors = ['#5C3D2E', '#B96420', '#F5EFE6', '#8B7355']

# 图1: 模型对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('不同模型的性能对比', fontsize=14, fontweight='bold')

model_names = list(results.keys()) + ['鲁棒模型']
valid_maes = [results[k]['valid_mae'] for k in results.keys()] + [valid_mae_robust]
test_maes = [results[k]['test_mae'] for k in results.keys()] + [test_mae_robust]

x = np.arange(len(model_names))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2, valid_maes, width, label='验证集', color=colors[0])
bars2 = ax.bar(x + width/2, test_maes, width, label='测试集', color=colors[1])
ax.set_xlabel('模型', fontsize=11)
ax.set_ylabel('MAE', fontsize=11)
ax.set_title('(a) 各模型MAE对比', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 测试集预测散点图（使用鲁棒模型）
ax = axes[1]
ax.scatter(y_test, y_test_pred_robust, alpha=0.5, color=colors[1], s=20)
ax.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='理想预测线')
ax.set_xlabel('真实标签', fontsize=11)
ax.set_ylabel('预测值', fontsize=11)
ax.set_title(f'(b) 鲁棒模型预测 (MAE={test_mae_robust:.4f})', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.2, 3.2)

plt.tight_layout()
plt.savefig(fig_dir / 'problem2_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem2_model_comparison.png'}")

# 图2: 附件3预测结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('附件3模态缺失样本预测结果', fontsize=14, fontweight='bold')

sample_ids = missing_pred_df['样本编号'].values
predictions = missing_pred_df['预测值'].values

# (a) 预测值条形图
ax = axes[0]
bars = ax.bar(sample_ids, predictions, color=colors[0], edgecolor='black', alpha=0.8, width=0.8)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('样本编号', fontsize=11)
ax.set_ylabel('预测情感值', fontsize=11)
ax.set_title('(a) 30个缺失样本的预测情感值', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(range(1, 31, 2))

# 标记极端值
for sid, pred in zip(sample_ids, predictions):
    if abs(pred) > 1.5:
        ax.text(sid, pred, f'{pred:.2f}', ha='center',
                va='bottom' if pred > 0 else 'top', fontsize=8)

# (b) 零值占比热图
ax = axes[1]
modality_zeros = missing_pred_df[['text零值占比(%)', 'vision零值占比(%)', 'audio零值占比(%)']].values.T
im = ax.imshow(modality_zeros, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=100)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['文本', '视觉', '音频'])
ax.set_xlabel('样本编号', fontsize=11)
ax.set_title('(b) 各模态零值占比热图 (%)', fontsize=12)
ax.set_xticks(range(0, 30, 2))
ax.set_xticklabels(range(1, 31, 2))
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('零值占比 (%)', fontsize=10)

plt.tight_layout()
plt.savefig(fig_dir / 'problem2_missing_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem2_missing_predictions.png'}")

# 图3: 特征重要性（来自最佳基础模型）
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_

    # 按模态聚合重要性
    text_end = 18
    vision_end = 228
    audio_end = 672

    text_importance = importances[:text_end].sum()
    vision_importance = importances[text_end:vision_end].sum()
    audio_importance = importances[vision_end:audio_end].sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    modalities = ['文本', '视觉', '音频']
    modal_importances = [text_importance, vision_importance, audio_importance]

    bars = ax.bar(modalities, modal_importances, color=colors[:3], edgecolor='black', alpha=0.8)
    ax.set_ylabel('特征重要性', fontsize=12)
    ax.set_title('各模态特征的重要性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, imp in zip(bars, modal_importances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(fig_dir / 'problem2_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {fig_dir / 'problem2_feature_importance.png'}")

# ========== 7. 性能统计 ==========
print("\n[7/7] 生成性能统计...")

performance_stats = pd.DataFrame({
    '模型': list(results.keys()) + ['鲁棒模型'],
    '验证MAE': [results[k]['valid_mae'] for k in results.keys()] + [valid_mae_robust],
    '测试MAE': [results[k]['test_mae'] for k in results.keys()] + [test_mae_robust],
    '测试RMSE': [results[k]['test_rmse'] for k in results.keys()] + [np.sqrt(mean_squared_error(y_test, y_test_pred_robust))],
    '测试R²': [results[k]['test_r2'] for k in results.keys()] + [r2_score(y_test, y_test_pred_robust)]
})

performance_stats.to_csv(output_dir / 'performance_summary.csv', index=False, encoding='utf-8-sig')
print(f"\n性能统计:")
try:
    print(performance_stats.to_string(index=False))
except UnicodeEncodeError:
    # Windows控制台编码问题，改用简化输出
    for idx, row in performance_stats.iterrows():
        print(f"{row['模型']}: Valid MAE={row['验证MAE']:.4f}, Test MAE={row['测试MAE']:.4f}, Test R2={row['测试R²']:.4f}")


print("\n" + "=" * 60)
print("问题2完成！")
print("=" * 60)
print(f"\n结果文件保存在: {output_dir}")
print(f"图表文件保存在: {fig_dir}")
print(f"模型文件保存在: {model_dir}")
