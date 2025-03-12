import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # 진행 상태 표시

# 데이터셋 폴더 경로 (오디오 파일이 들어 있는 폴더)
DATASET_PATH = "C:/Users/u0102/Desktop/BIAS/AI/Data/genres_original"
FEATURES_CSV = "C:/Users/u0102/Desktop/BIAS/AI/guitar_features.csv"

# 특징 추출 함수
def extract_features(file_path, sample_rate=22050):
    y, sr = librosa.load(file_path, sr=sample_rate)

    # 오디오 특징 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCC (13개 계수)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Chromagram
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # Spectral Contrast
    zero_crossing = librosa.feature.zero_crossing_rate(y)  # Zero Crossing Rate
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # 템포 감지 (BPM)

    # 평균값을 사용하여 벡터 변환
    features = np.hstack([
        np.mean(mfcc, axis=1), 
        np.mean(chroma, axis=1), 
        np.mean(spec_contrast, axis=1), 
        np.mean(zero_crossing),
        tempo
    ])
    
    return features

# 데이터셋에서 특징 추출 및 저장
features = []
labels = []
genres = os.listdir(DATASET_PATH)

for genre in tqdm(genres):  # 진행 상태 표시
    genre_path = os.path.join(DATASET_PATH, genre)
    if not os.path.isdir(genre_path):  # 폴더인지 확인
        continue
    
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        try:
            feature_vector = extract_features(file_path)
            features.append(feature_vector)
            labels.append(genre)  # 해당 파일이 속한 장르 저장
        except Exception as e:
            print(f"❌ {file} 처리 중 오류 발생: {e}")

# Pandas DataFrame으로 저장
df = pd.DataFrame(features, columns=[
    *[f"mfcc_{i}" for i in range(13)], 
    *[f"chroma_{i}" for i in range(12)], 
    *[f"spec_contrast_{i}" for i in range(7)], 
    "zero_crossing", "tempo"
])
df["label"] = labels  # 장르 정보 추가
df.to_csv(FEATURES_CSV, index=False)
print(f"✅ 특징이 {FEATURES_CSV}에 저장되었습니다!")
