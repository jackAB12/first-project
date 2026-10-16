#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本程序及代码是在人工智能工具辅助下完成的，并标明人工智能工具名称，版本/型号，开发机构/公司，版本颁布日期
人工智能工具名称：Claude
版本:Claude Fable5
开发机构:Authropic
版本颁布日期:2026年6月9日

问题1：从100条原始视频提取多模态特征并对齐

使用ModelScope国内镜像源下载BERT模型
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置ModelScope镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 视频和音频处理
try:
    import cv2
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"警告：缺少依赖库 {e}")
    print("请安装：pip install opencv-python librosa soundfile")

# 文本编码
try:
    from modelscope import snapshot_download
    from transformers import BertTokenizer, BertModel
    import torch
except ImportError:
    print("警告：缺少transformers或modelscope库")
    print("请安装：pip install transformers torch modelscope")

print("="*70)
print("问题1：多模态特征提取与时序对齐")
print("="*70)

# ============================================================
# 配置
# ============================================================

VIDEO_DIR = Path("../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条")
LABEL_FILE = VIDEO_DIR / "label-100.xlsx"
OUTPUT_DIR = Path("./results/problem1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 50
TEXT_DIM = 768
AUDIO_DIM = 74
VISION_DIM = 35

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# ============================================================
# 1. 文本特征提取
# ============================================================

def extract_text_features(text, bert_model, tokenizer, max_len=50):
    """从文本提取BERT特征"""
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_features = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    text_bert = np.stack([
        input_ids.squeeze(0).cpu().numpy(),
        attention_mask.squeeze(0).cpu().numpy(),
        token_type_ids.squeeze(0).cpu().numpy()
    ], axis=0)

    return text_features, text_bert

# ============================================================
# 2. 语音特征提取
# ============================================================

def extract_audio_from_video(video_path, output_audio_path=None):
    """从视频提取音频"""
    if output_audio_path is None:
        output_audio_path = video_path.with_suffix('.wav')

    cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_audio_path}" -y -loglevel quiet'
    os.system(cmd)

    return output_audio_path

def extract_audio_features(audio_path, max_len=50, target_dim=74):
    """从音频文件提取声学特征"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        # 提取多种声学特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        features = np.vstack([
            mfcc, mel_db, chroma, spectral_centroid,
            spectral_bandwidth, spectral_contrast, zcr, rms
        ])

        if features.shape[0] < target_dim:
            padding = np.zeros((target_dim - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        elif features.shape[0] > target_dim:
            features = features[:target_dim, :]

        features = features.T
        features_aligned = align_temporal_sequence(features, max_len)

        return features_aligned

    except Exception as e:
        print(f"  音频特征提取失败: {e}")
        return np.zeros((max_len, target_dim))

# ============================================================
# 3. 视觉特征提取
# ============================================================

def extract_vision_features(video_path, max_len=50, target_dim=35):
    """从视频提取视觉特征"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            std = np.std(gray)
            hist, _ = np.histogram(gray, bins=33)
            hist = hist / hist.sum()

            features = np.concatenate([[mean, std], hist])
            frame_features.append(features)

        cap.release()

        if len(frame_features) == 0:
            return np.zeros((max_len, target_dim))

        frame_features = np.array(frame_features)
        features_aligned = align_temporal_sequence(frame_features, max_len)

        return features_aligned

    except Exception as e:
        print(f"  视觉特征提取失败: {e}")
        return np.zeros((max_len, target_dim))

# ============================================================
# 4. 时序对齐
# ============================================================

def align_temporal_sequence(features, target_len):
    """将时序特征对齐到目标长度"""
    T, D = features.shape

    if T == target_len:
        return features
    elif T < target_len:
        padding = np.zeros((target_len - T, D))
        return np.vstack([features, padding])
    else:
        indices = np.linspace(0, T - 1, target_len, dtype=int)
        return features[indices]

# ============================================================
# 5. 主处理流程
# ============================================================

def process_video_sample(video_id, clip_id, text, label, annotation,
                        bert_model, tokenizer):
    """处理单个视频样本"""
    video_path = VIDEO_DIR / video_id / f"{clip_id}.mp4"

    if not video_path.exists():
        print(f"  警告：视频文件不存在 {video_path}")
        return None, None

    print(f"  处理视频: {video_id}/{clip_id}.mp4")

    # 获取视频时长
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    # 1. 提取文本特征
    text_features, text_bert = extract_text_features(text, bert_model, tokenizer)

    # 2. 提取音频特征
    audio_path = OUTPUT_DIR / "temp_audio" / f"{video_id}_{clip_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    extract_audio_from_video(video_path, audio_path)
    audio_features = extract_audio_features(audio_path, MAX_SEQ_LEN, AUDIO_DIM)

    # 3. 提取视觉特征
    vision_features = extract_vision_features(video_path, MAX_SEQ_LEN, VISION_DIM)

    # 组织样本数据
    sample_data = {
        'id': f"{video_id}$_${clip_id}",
        'raw_text': text,
        'text': text_features,
        'text_bert': text_bert,
        'audio': audio_features,
        'vision': vision_features,
        'classification_labels': 0 if annotation == 'Neutral' else (1 if annotation == 'Positive' else -1),
        'regression_labels': float(label),
        'annotations': annotation
    }

    sample_info = {
        'sample_id': f"{video_id}$_${clip_id}",
        'video_id': video_id,
        'clip_id': clip_id,
        'duration_sec': duration,
        'text_dim': text_features.shape,
        'audio_dim': audio_features.shape,
        'vision_dim': vision_features.shape,
        'aligned_len': MAX_SEQ_LEN,
        'label': label,
        'annotation': annotation
    }

    return sample_data, sample_info

def main():
    """主函数"""

    print("\n[1/5] 使用ModelScope镜像源加载BERT模型...")

    try:
        # 使用ModelScope下载模型
        model_dir = snapshot_download('AI-ModelScope/bert-base-uncased',
                                     cache_dir='./models')

        tokenizer = BertTokenizer.from_pretrained(model_dir)
        bert_model = BertModel.from_pretrained(model_dir).to(device)
        bert_model.eval()
        print("✓ BERT模型加载成功")
    except Exception as e:
        print(f"✗ BERT模型加载失败: {e}")
        print("  尝试使用transformers直接加载...")
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
            bert_model.eval()
            print("✓ BERT模型加载成功（备用方案）")
        except Exception as e2:
            print(f"✗ 备用方案也失败: {e2}")
            return

    print("\n[2/5] 读取标注文件...")
    labels_df = pd.read_excel(LABEL_FILE)
    print(f"✓ 共{len(labels_df)}条样本")

    print("\n[3/5] 开始提取特征...")

    all_samples = []
    all_info = []

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="提取特征"):
        video_id = row['video_id']
        clip_id = int(row['clip_id'])
        text = row['text']
        label = row['label']
        annotation = row['annotation']

        sample_data, sample_info = process_video_sample(
            video_id, clip_id, text, label, annotation,
            bert_model, tokenizer
        )

        if sample_data is not None:
            all_samples.append(sample_data)
            all_info.append(sample_info)

    print(f"\n✓ 成功提取 {len(all_samples)}/100 个样本的特征")

    print("\n[4/5] 保存特征文件...")

    feature_data = {
        'train': {
            'id': np.array([s['id'] for s in all_samples]),
            'raw_text': np.array([s['raw_text'] for s in all_samples]),
            'text': np.stack([s['text'] for s in all_samples]),
            'text_bert': np.stack([s['text_bert'] for s in all_samples]),
            'audio': np.stack([s['audio'] for s in all_samples]),
            'vision': np.stack([s['vision'] for s in all_samples]),
            'classification_labels': np.array([s['classification_labels'] for s in all_samples]),
            'regression_labels': np.array([s['regression_labels'] for s in all_samples]),
            'annotations': np.array([s['annotations'] for s in all_samples])
        }
    }

    output_pkl = OUTPUT_DIR / "extracted_100_features.pkl"
    with open(output_pkl, 'wb') as f:
        pickle.dump(feature_data, f)
    print(f"✓ 特征文件已保存: {output_pkl}")

    print("\n[5/5] 生成汇总表...")

    info_df = pd.DataFrame(all_info)
    info_csv = OUTPUT_DIR / "feature_extraction_summary.csv"
    info_df.to_csv(info_csv, index=False, encoding='utf-8-sig')
    print(f"✓ 汇总表已保存: {info_csv}")

    print("\n" + "="*70)
    print("特征提取完成！")
    print("="*70)
    print(f"特征文件: {output_pkl}")
    print(f"汇总表: {info_csv}")
    print(f"成功处理: {len(all_samples)}/100 个样本")

if __name__ == "__main__":
    main()
