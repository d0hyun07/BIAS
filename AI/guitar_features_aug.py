import os
import pandas as pd
import numpy as np
import librosa
import librosa.display

# 📌 1️⃣ 오디오 특징 추출 함수 (Zero Crossing Rate, 템포 포함)
def extract_features(y, sr):
    # 최소 길이 체크 (너무 짧으면 제외)
    if len(y) < 4096:
        print("⚠️ 오디오가 너무 짧아서 제외됨.")
        return None

    try:
        # MFCC (음색 특징)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Chromagram (화음 특징) - 오류 방지
        if len(y) > 4096:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        else:
            chroma = np.zeros((12, 1))

        # Spectral Contrast (주파수 대비) - 오류 방지
        if len(y) > 8192:
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        else:
            spec_contrast = np.zeros((7, 1))

        # Zero Crossing Rate (리듬 분석)
        zero_crossing = librosa.feature.zero_crossing_rate(y)

        # 템포 (BPM 분석)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # 특징 벡터 변환 (평균값 사용)
        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spec_contrast, axis=1),
            np.mean(zero_crossing),
            tempo
        ])
        
        # NaN 값 검사
        if np.isnan(features).any():
            print("⚠️ NaN 값 발견! 데이터 제외됨.")
            return None
        
        return features

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")
        return None

# 📌 2️⃣ 오디오 증강 (Data Augmentation)
def augment_audio(y, sr):
    augmented = []
    try:
        # 랜덤 피치 변형 (Pitch Shifting)
        pitch_shift = np.random.uniform(-2, 2)
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
        if len(y_pitch) > 4096:
            augmented.append(y_pitch)

        # 랜덤 타임 스트레칭 (Time Stretching) - 길이 유지
        stretch_rate = np.random.uniform(0.8, 1.2)
        y_stretch = librosa.effects.time_stretch(y, rate=stretch_rate)
        if len(y_stretch) > 4096:
            augmented.append(y_stretch)

        # 랜덤 잡음 추가 (Adding Noise)
        noise_factor = 0.005 * np.random.randn(len(y))
        y_noise = y + noise_factor
        augmented.append(y_noise)

    except Exception as e:
        print(f"⚠️ 증강 중 오류 발생: {e}")

    return augmented

# 📌 3️⃣ 데이터셋 로드 및 특징 추출
dataset_path = "C:/Users/u0102/Desktop/BIAS/AI/Data/genres_original"  # 오디오 파일이 저장된 폴더
data = []
labels = []

for genre in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre)
    
    if not os.path.isdir(genre_path):  # 폴더가 아니면 스킵
        continue

    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)

        # 오디오 파일 로드
        try:
            y, sr = librosa.load(file_path, sr=22050)
        except Exception as e:
            print(f"⚠️ {file} 파일 로드 중 오류 발생: {e}")
            continue
        
        # 원본 데이터 저장
        features = extract_features(y, sr)
        if features is not None:
            data.append(features)
            labels.append(genre)

        # 증강된 데이터 저장
        for aug in augment_audio(y, sr):
            aug_features = extract_features(aug, sr)
            if aug_features is not None:
                data.append(aug_features)
                labels.append(genre)

# 📌 4️⃣ DataFrame으로 변환 후 CSV 저장
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv("guitar_features_aug.csv", index=False)

print(f"🔥 데이터 전처리 완료! 총 {len(df)}개의 샘플 저장됨.")
