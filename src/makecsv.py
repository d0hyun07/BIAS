import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

base_dir = "C:/Users/u0102/Desktop/BIAS/data/raw_audio"
genres = ['funk', 'jazz', 'rock', 'pop']
tempos = ['slow', 'middle', 'fast']

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spec_contrast, axis=1),
        np.mean(rolloff),
        np.mean(bandwidth),
        np.mean(tonnetz, axis=1),
        np.mean(zero_crossing),
        tempo
    ])

data = []

for genre in genres:
    for tempo in tempos:
        folder = os.path.join(base_dir, genre, tempo)
        for file in tqdm(os.listdir(folder)):
            if file.endswith('.wav'):
                file_path = os.path.join(folder, file)
                y, sr = librosa.load(file_path, sr=22050)
                features = extract_features(y, sr)
                data.append([*features, genre, tempo, file])

# 컬럼명 추가 (확장된 버전)
columns = [f"mfcc_{i}" for i in range(13)] + \
          [f"chroma_{i}" for i in range(12)] + \
          [f"spec_contrast_{i}" for i in range(7)] + \
          ["rolloff", "bandwidth"] + \
          [f"tonnetz_{i}" for i in range(6)] + \
          ["zero_crossing", "tempo", "genre", "tempo_label", "file_name"]

# CSV로 저장
df = pd.DataFrame(data, columns=columns)
df.to_csv("augmented_dataset.csv", index=False)
print("✅ 확장된 특징으로 CSV 저장 완료!")
