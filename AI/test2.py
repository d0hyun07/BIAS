import librosa
import numpy as np
import pandas as pd

# 샘플 오디오 파일 로드 (기타 데이터로 변경 가능)
file_path = librosa.example("nutcracker")  # 샘플 오디오
y, sr = librosa.load(file_path, sr=22050)

# 오디오 특징 추출
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCC
chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Chromagram
spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # Spectral Contrast
zero_crossing = librosa.feature.zero_crossing_rate(y)  # Zero Crossing Rate

# 특징 벡터를 데이터프레임으로 저장
features = np.hstack([
    np.mean(mfcc, axis=1), 
    np.mean(chroma, axis=1), 
    np.mean(spec_contrast, axis=1), 
    np.mean(zero_crossing)
])

df = pd.DataFrame([features], columns=[
    *[f"mfcc_{i}" for i in range(13)], 
    *[f"chroma_{i}" for i in range(12)], 
    *[f"spec_contrast_{i}" for i in range(7)], 
    "zero_crossing"
])

# 데이터 출력
print(df.head())
