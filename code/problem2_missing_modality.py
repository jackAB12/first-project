"""
问题2：模态缺失下的情感预测
任务：构建Transformer + 模态融合模型，处理模态缺失情况

方法：
1. 使用Transformer编码器提取每个模态的时序特征
2. 使用注意力机制融合多模态特征
3. 训练时随机mask模态模拟缺失，提升鲁棒性
4. 测试时对附件3的30个缺失样本进行预测
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', palette='Set2')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

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
print("\n[1/8] 加载数据...")

with open('E题数据/附件2-数据集特征文件/aligned_50.pkl', 'rb') as f:
    aligned_data = pickle.load(f)

train_data = aligned_data['train']
valid_data = aligned_data['valid']
test_data = aligned_data['test']

print(f"训练集: {train_data['audio'].shape[0]} 样本")
print(f"验证集: {valid_data['audio'].shape[0]} 样本")
print(f"测试集: {test_data['audio'].shape[0]} 样本")

# ========== 2. 数据集类 ==========
class MultimodalDataset(Dataset):
    """多模态数据集，支持训练时随机模态缺失"""

    def __init__(self, data, missing_prob=0.0, mode='train'):
        """
        Args:
            data: 数据字典
            missing_prob: 训练时随机缺失某个模态的概率
            mode: 'train', 'valid', 'test'
        """
        self.text = torch.FloatTensor(data['text_bert'])  # (N, 3, 50)
        self.vision = torch.FloatTensor(data['vision'])    # (N, 50, 35)
        self.audio = torch.FloatTensor(data['audio'])      # (N, 50, 74)
        self.labels = torch.FloatTensor(data['regression_labels'])  # (N,)

        self.missing_prob = missing_prob
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.text[idx]      # (3, 50)
        vision = self.vision[idx]  # (50, 35)
        audio = self.audio[idx]    # (50, 74)
        label = self.labels[idx]

        # 创建mask标记（1表示存在，0表示缺失）
        text_mask = torch.ones(1)
        vision_mask = torch.ones(1)
        audio_mask = torch.ones(1)

        # 训练时随机mask某些模态
        if self.mode == 'train' and self.missing_prob > 0:
            if np.random.rand() < self.missing_prob:
                text_mask = torch.zeros(1)
                text = torch.zeros_like(text)
            if np.random.rand() < self.missing_prob:
                vision_mask = torch.zeros(1)
                vision = torch.zeros_like(vision)
            if np.random.rand() < self.missing_prob:
                audio_mask = torch.zeros(1)
                audio = torch.zeros_like(audio)

        return {
            'text': text,
            'vision': vision,
            'audio': audio,
            'text_mask': text_mask,
            'vision_mask': vision_mask,
            'audio_mask': audio_mask,
            'label': label
        }

# ========== 3. 模型定义 ==========
class ModalityEncoder(nn.Module):
    """单模态Transformer编码器"""

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = hidden_dim

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, 1) 模态是否存在的mask
        Returns:
            encoded: (batch, hidden_dim) 编码后的特征
        """
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)

        # 使用平均池化得到固定长度表示
        x = x.mean(dim=1)  # (batch, hidden_dim)

        # 如果模态缺失，则输出全零
        if mask is not None:
            x = x * mask

        return x


class MultimodalFusionModel(nn.Module):
    """多模态融合模型，使用注意力机制"""

    def __init__(self, text_dim=50, vision_dim=35, audio_dim=74,
                 hidden_dim=128, fusion_dim=256, dropout=0.2):
        super().__init__()

        # 三个模态的编码器
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, num_heads=4, num_layers=2)
        self.vision_encoder = ModalityEncoder(vision_dim, hidden_dim, num_heads=4, num_layers=2)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim, num_heads=4, num_layers=2)

        # 注意力融合
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)

        # 融合后的MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1)
        )

    def forward(self, text, vision, audio, text_mask=None, vision_mask=None, audio_mask=None):
        """
        Args:
            text: (batch, 3, 50)
            vision: (batch, 50, 35)
            audio: (batch, 50, 74)
            *_mask: (batch, 1) 模态是否存在
        Returns:
            output: (batch, 1) 情感预测值
        """
        batch_size = text.size(0)

        # 转置text以匹配(batch, seq, feat)格式
        text = text.transpose(1, 2)  # (batch, 50, 3)

        # 编码每个模态
        text_feat = self.text_encoder(text, text_mask)      # (batch, hidden_dim)
        vision_feat = self.vision_encoder(vision, vision_mask)  # (batch, hidden_dim)
        audio_feat = self.audio_encoder(audio, audio_mask)      # (batch, hidden_dim)

        # 堆叠为(batch, 3, hidden_dim)用于attention
        modality_feats = torch.stack([text_feat, vision_feat, audio_feat], dim=1)

        # 创建key_padding_mask (True表示忽略该位置)
        # mask: (batch, 3)，True表示该模态缺失
        if text_mask is not None:
            key_padding_mask = torch.cat([
                (text_mask < 0.5),
                (vision_mask < 0.5),
                (audio_mask < 0.5)
            ], dim=1)  # (batch, 3)
        else:
            key_padding_mask = None

        # 自注意力融合
        fused_feat, attn_weights = self.attention(
            modality_feats, modality_feats, modality_feats,
            key_padding_mask=key_padding_mask
        )  # fused_feat: (batch, 3, hidden_dim)

        # 平均池化
        fused_feat = fused_feat.mean(dim=1)  # (batch, hidden_dim)

        # 回归预测
        output = self.fusion_mlp(fused_feat)  # (batch, 1)

        return output.squeeze(-1), attn_weights


# ========== 4. 训练函数 ==========
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Training', leave=False):
        text = batch['text'].to(device)
        vision = batch['vision'].to(device)
        audio = batch['audio'].to(device)
        text_mask = batch['text_mask'].to(device)
        vision_mask = batch['vision_mask'].to(device)
        audio_mask = batch['audio_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        preds, _ = model(text, vision, audio, text_mask, vision_mask, audio_mask)
        loss = criterion(preds, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

    return avg_loss, mae


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            text = batch['text'].to(device)
            vision = batch['vision'].to(device)
            audio = batch['audio'].to(device)
            text_mask = batch['text_mask'].to(device)
            vision_mask = batch['vision_mask'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            labels = batch['label'].to(device)

            preds, _ = model(text, vision, audio, text_mask, vision_mask, audio_mask)
            loss = criterion(preds, labels)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

    return avg_loss, mae, all_preds, all_labels


# ========== 5. 创建数据加载器 ==========
print("\n[2/8] 创建数据加载器...")

# 使用0.15的缺失概率训练，提升鲁棒性
train_dataset = MultimodalDataset(train_data, missing_prob=0.15, mode='train')
valid_dataset = MultimodalDataset(valid_data, missing_prob=0.0, mode='valid')

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(valid_loader)}")

# ========== 6. 训练模型 ==========
print("\n[3/8] 训练模型...")

model = MultimodalFusionModel(
    text_dim=50,
    vision_dim=35,
    audio_dim=74,
    hidden_dim=128,
    fusion_dim=256,
    dropout=0.2
).to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

num_epochs = 50
best_val_mae = float('inf')
history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

for epoch in range(num_epochs):
    train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_mae, _, _ = evaluate(model, valid_loader, criterion, device)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_mae'].append(train_mae)
    history['val_loss'].append(val_loss)
    history['val_mae'].append(val_mae)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f} | "
          f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

    # 保存最佳模型
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), model_dir / 'problem2_best_model.pt')
        print(f"  -> 保存最佳模型 (MAE={val_mae:.4f})")

print(f"\n最佳验证MAE: {best_val_mae:.4f}")

# 保存训练历史
history_df = pd.DataFrame(history)
history_df.to_csv(output_dir / 'training_history.csv', index=False)

# ========== 7. 加载最佳模型并测试 ==========
print("\n[4/8] 测试最佳模型...")

model.load_state_dict(torch.load(model_dir / 'problem2_best_model.pt'))

# 测试集
test_dataset = MultimodalDataset(test_data, missing_prob=0.0, mode='test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_loss, test_mae, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
print(f"测试集 MAE: {test_mae:.4f}")

# 保存测试结果
test_results = pd.DataFrame({
    'true_label': test_labels,
    'predicted': test_preds,
    'error': np.abs(np.array(test_preds) - np.array(test_labels))
})
test_results.to_csv(output_dir / 'test_predictions.csv', index=False)

# ========== 8. 预测附件3的模态缺失样本 ==========
print("\n[5/8] 预测附件3的30个模态缺失样本...")

missing_predictions = []

for i in range(1, 31):
    with open(f'E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{i:02d}.pkl', 'rb') as f:
        missing_data = pickle.load(f)

    missing_test = missing_data['test']

    # 转换为tensor
    text = torch.FloatTensor(missing_test['text_bert']).to(device)  # (1, 3, 50)
    vision = torch.FloatTensor(missing_test['vision']).to(device)    # (1, 50, 35)
    audio = torch.FloatTensor(missing_test['audio']).to(device)      # (1, 50, 74)

    # 检测每个模态是否有效（非全零或接近全零）
    text_valid = (torch.abs(text).sum() > 1e-3).float().unsqueeze(0)
    vision_valid = (torch.abs(vision).sum() > 1e-3).float().unsqueeze(0)
    audio_valid = (torch.abs(audio).sum() > 1e-3).float().unsqueeze(0)

    # 统计零值占比来更精确判断缺失
    text_zero_pct = (text == 0).sum().item() / text.numel() * 100
    vision_zero_pct = (vision == 0).sum().item() / vision.numel() * 100
    audio_zero_pct = (audio == 0).sum().item() / audio.numel() * 100

    # 预测
    model.eval()
    with torch.no_grad():
        pred, attn_weights = model(text, vision, audio, text_valid, vision_valid, audio_valid)

    pred_value = pred.item()
    missing_predictions.append({
        '样本编号': i,
        '预测值': pred_value,
        'text零值占比(%)': text_zero_pct,
        'vision零值占比(%)': vision_zero_pct,
        'audio零值占比(%)': audio_zero_pct
    })

    print(f"附件3_{i:02d}: 预测={pred_value:.4f}, text零值={text_zero_pct:.1f}%, vision零值={vision_zero_pct:.1f}%, audio零值={audio_zero_pct:.1f}%")

# 保存预测结果
missing_pred_df = pd.DataFrame(missing_predictions)
missing_pred_df.to_csv(output_dir / 'missing_modality_predictions.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存附件3预测结果: {output_dir / 'missing_modality_predictions.csv'}")

# ========== 9. 可视化训练过程 ==========
print("\n[6/8] 生成训练过程可视化...")

colors = ['#5C3D2E', '#B96420', '#F5EFE6']

# 图1: 训练曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('模型训练过程', fontsize=14, fontweight='bold')

epochs = range(1, num_epochs + 1)

# (a) Loss曲线
ax = axes[0]
ax.plot(epochs, history['train_loss'], label='训练集', color=colors[0], linewidth=2)
ax.plot(epochs, history['val_loss'], label='验证集', color=colors[1], linewidth=2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('MSE Loss', fontsize=11)
ax.set_title('(a) 损失函数曲线', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# (b) MAE曲线
ax = axes[1]
ax.plot(epochs, history['train_mae'], label='训练集', color=colors[0], linewidth=2)
ax.plot(epochs, history['val_mae'], label='验证集', color=colors[1], linewidth=2)
ax.axhline(best_val_mae, color='red', linestyle='--', label=f'最佳MAE={best_val_mae:.4f}')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('MAE', fontsize=11)
ax.set_title('(b) 平均绝对误差曲线', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'problem2_training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem2_training_curves.png'}")

# 图2: 测试集预测结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('测试集预测效果', fontsize=14, fontweight='bold')

# (a) 散点图
ax = axes[0]
ax.scatter(test_labels, test_preds, alpha=0.5, color=colors[0], s=20)
ax.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='理想预测线')
ax.set_xlabel('真实标签', fontsize=11)
ax.set_ylabel('预测值', fontsize=11)
ax.set_title(f'(a) 真实值 vs 预测值 (MAE={test_mae:.4f})', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.2, 3.2)

# (b) 误差分布
ax = axes[1]
errors = np.array(test_preds) - np.array(test_labels)
ax.hist(errors, bins=40, color=colors[1], edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差')
ax.set_xlabel('预测误差', fontsize=11)
ax.set_ylabel('样本数', fontsize=11)
ax.set_title('(b) 预测误差分布', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(fig_dir / 'problem2_test_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem2_test_performance.png'}")

# 图3: 附件3预测结果可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('附件3模态缺失样本预测结果', fontsize=14, fontweight='bold')

sample_ids = missing_pred_df['样本编号'].values
predictions = missing_pred_df['预测值'].values

# (a) 预测值分布
ax = axes[0]
bars = ax.bar(sample_ids, predictions, color=colors[0], edgecolor='black', alpha=0.8)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('样本编号', fontsize=11)
ax.set_ylabel('预测情感值', fontsize=11)
ax.set_title('(a) 30个缺失样本的预测情感值', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(range(1, 31, 2))

# 标记极端值
for i, (sid, pred) in enumerate(zip(sample_ids, predictions)):
    if abs(pred) > 1.5:
        ax.text(sid, pred, f'{pred:.2f}', ha='center', va='bottom' if pred > 0 else 'top', fontsize=8)

# (b) 模态零值占比热图
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

# ========== 10. 模型性能统计 ==========
print("\n[7/8] 生成性能统计...")

performance_stats = {
    '指标': ['训练集MAE', '验证集MAE', '测试集MAE', '附件3预测均值', '附件3预测标准差'],
    '数值': [
        history['train_mae'][-1],
        best_val_mae,
        test_mae,
        np.mean(predictions),
        np.std(predictions)
    ]
}

perf_df = pd.DataFrame(performance_stats)
perf_df.to_csv(output_dir / 'performance_summary.csv', index=False, encoding='utf-8-sig')
print(f"\n性能统计:")
print(perf_df.to_string(index=False))

# ========== 11. 鲁棒性分析：不同缺失率的影响 ==========
print("\n[8/8] 评估不同缺失率下的模型性能...")

missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
robustness_results = []

model.eval()
for miss_rate in missing_rates:
    temp_dataset = MultimodalDataset(valid_data, missing_prob=miss_rate, mode='valid')
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    _, mae, _, _ = evaluate(model, temp_loader, criterion, device)
    robustness_results.append({
        '缺失率': miss_rate,
        'MAE': mae
    })
    print(f"缺失率 {miss_rate:.1%}: MAE = {mae:.4f}")

robustness_df = pd.DataFrame(robustness_results)
robustness_df.to_csv(output_dir / 'robustness_analysis.csv', index=False, encoding='utf-8-sig')

# 可视化鲁棒性分析
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(robustness_df['缺失率'], robustness_df['MAE'], marker='o', linewidth=2,
        markersize=8, color=colors[1])
ax.set_xlabel('模态缺失率', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('模型在不同缺失率下的鲁棒性分析', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(missing_rates)
ax.set_xticklabels([f'{int(r*100)}%' for r in missing_rates])

# 标注数值
for i, (rate, mae) in enumerate(zip(robustness_df['缺失率'], robustness_df['MAE'])):
    ax.text(rate, mae, f'{mae:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(fig_dir / 'problem2_robustness_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_dir / 'problem2_robustness_analysis.png'}")

print("\n" + "=" * 60)
print("问题2完成！")
print("=" * 60)
print(f"\n结果文件保存在: {output_dir}")
print(f"图表文件保存在: {fig_dir}")
print(f"模型文件保存在: {model_dir}")

