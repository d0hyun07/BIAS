import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization

# 📌 1️⃣ 오디오 증강 (Data Augmentation)
def augment_audio(y, sr):
    # 랜덤 피치 변형 (Pitch Shifting)
    pitch_shift = np.random.uniform(-2, 2)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

    # 랜덤 타임 스트레칭 (Time Stretching)
    stretch_rate = np.random.uniform(0.8, 1.2)
    y_stretch = librosa.effects.time_stretch(y, rate=stretch_rate)

    # 랜덤 잡음 추가 (Adding Noise)
    noise_factor = 0.005 * np.random.randn(len(y))
    y_noise = y + noise_factor

    return [y_pitch, y_stretch, y_noise]

# 📌 2️⃣ 기존 `guitar_features.csv` 불러오기
df = pd.read_csv("guitar_features.csv")

# 증강된 데이터를 저장할 리스트
new_features = []
new_labels = []

for i, row in df.iterrows():
    features = row[:-1].values  # 기존 특징 사용
    label = row['label']
    
    # 원본 데이터 추가
    new_features.append(features)
    new_labels.append(label)

    # 증강된 데이터 생성
    y, sr = librosa.load(row['file_path'], sr=22050)  # 파일 경로에서 로드
    for aug in augment_audio(y, sr):
        aug_features = librosa.feature.mfcc(y=aug, sr=sr, n_mfcc=13).mean(axis=1)
        new_features.append(aug_features)
        new_labels.append(label)

# 📌 3️⃣ 데이터 전처리 (라벨 인코딩 & 정규화)
X = np.array(new_features)
y = np.array(new_labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 데이터 형태 변환 (CNN + LSTM 입력 맞추기)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"🔥 데이터셋 크기 (증강 포함): {X_train.shape[0]} 개의 훈련 샘플, {X_test.shape[0]} 개의 테스트 샘플")

# 📌 4️⃣ CNN + LSTM 모델 학습
model = Sequential([
    Conv1D(128, kernel_size=9, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),  
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    LSTM(512, return_sequences=True),
    LSTM(512, return_sequences=False),
    Dropout(0.4),

    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(len(encoder.classes_), activation='softmax')  
])

# 학습률 조정 (Adam Optimizer 개선)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

# 모델 컴파일
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"🔥 최적화된 CNN + LSTM 모델의 테스트 정확도: {accuracy:.4f}")
