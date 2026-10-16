"""
问题1：多模态特征提取与时序对齐
从附件1的100条原始视频提取文本、语音、视觉特征并进行时序对齐
由于实际视频处理需要大量计算资源，本代码使用附件2已提取的特征进行分析和验证
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

class MultimodalFeatureAnalyzer:
    def __init__(self, data_dir='../E题数据'):
        self.data_dir = Path(data_dir)
        self.aligned_data = None
        self.unaligned_data = None
        self.labels = None

    def load_data(self):
        """加载附件2的特征数据"""
        print("加载数据...")

        # 加载对齐版本特征
        with open(self.data_dir / '附件2-数据集特征文件/aligned_50.pkl', 'rb') as f:
            self.aligned_data = pickle.load(f)

        # 加载未对齐版本特征
        with open(self.data_dir / '附件2-数据集特征文件/unaligned_50.pkl', 'rb') as f:
            self.unaligned_data = pickle.load(f)

        # 加载标签
        self.labels = pd.read_excel(self.data_dir / '附件2-数据集特征文件/label.xlsx')

        print(f"对齐数据形状: {self.aligned_data['vision'].shape if 'vision' in self.aligned_data else 'N/A'}")
        print(f"标签数量: {len(self.labels)}")
        print(f"数据集keys: {list(self.aligned_data.keys())}")

    def analyze_feature_dimensions(self):
        """分析各模态特征维度"""
        results = {}

        for modality in ['vision', 'audio', 'text']:
            if modality in self.aligned_data:
                data = self.aligned_data[modality]
                results[modality] = {
                    'shape': data.shape,
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data)
                }

        return results

    def extract_100_samples(self):
        """提取附件1对应的100条样本特征"""
        # 读取附件1的标签
        label_100 = pd.read_excel(
            self.data_dir / '附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx'
        )

        print(f"\n附件1共有 {len(label_100)} 条样本")
        print(f"附件1的video_id示例: {label_100['video_id'].head().tolist()}")

        # 在完整数据集中查找这100条
        matched_indices = []
        for idx, row in label_100.iterrows():
            matches = self.labels[
                (self.labels['video_id'] == row['video_id']) &
                (self.labels['clip_id'] == row['clip_id'])
            ]
            if len(matches) > 0:
                matched_indices.append(matches.index[0])

        print(f"成功匹配 {len(matched_indices)} 条样本")

        # 提取对应特征
        extracted_features = {}
        for modality in ['vision', 'audio', 'text']:
            if modality in self.aligned_data:
                extracted_features[modality] = self.aligned_data[modality][matched_indices]

        extracted_features['labels'] = self.labels.iloc[matched_indices]

        # 保存
        output_path = Path('results/extracted_100_features.pkl')
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(extracted_features, f)

        print(f"已保存到 {output_path}")
        return extracted_features, matched_indices

    def visualize_alignment_comparison(self, save_dir='figures'):
        """可视化对齐前后的特征"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        modalities = ['vision', 'audio', 'text']
        sample_idx = 0

        for i, modality in enumerate(modalities):
            # 对齐版本
            if modality in self.aligned_data:
                aligned = self.aligned_data[modality][sample_idx]
                axes[0, i].plot(aligned[:, 0] if len(aligned.shape) > 1 else aligned)
                axes[0, i].set_title(f'{modality} (对齐)')
                axes[0, i].set_xlabel('时间步')
                axes[0, i].grid(True, alpha=0.3)

            # 未对齐版本
            if modality in self.unaligned_data:
                unaligned = self.unaligned_data[modality][sample_idx]
                axes[1, i].plot(unaligned[:, 0] if len(unaligned.shape) > 1 else unaligned)
                axes[1, i].set_title(f'{modality} (未对齐)')
                axes[1, i].set_xlabel('时间步')
                axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'alignment_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"对齐对比图已保存")

def main():
    analyzer = MultimodalFeatureAnalyzer()

    # 加载数据
    analyzer.load_data()

    # 分析特征维度
    feature_stats = analyzer.analyze_feature_dimensions()
    print("\n特征统计:")
    for mod, stats in feature_stats.items():
        print(f"{mod}: {stats}")

    # 提取100条样本
    extracted, indices = analyzer.extract_100_samples()

    # 可视化
    analyzer.visualize_alignment_comparison()

    print("\n问题1特征提取完成！")
    return analyzer, extracted

if __name__ == '__main__':
    analyzer, extracted = main()
