import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CSV 불러오기
df = pd.read_csv("C:/Users/u0102/Desktop/BIAS/data/features/augmented_dataset.csv")

# 데이터 분리
X = df.drop(columns=["genre", "tempo_label", "file_name"]).values
y = df["genre"].values

# 라벨 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 스케일링 (StandardScaler 사용)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # CNN 입력형태로 reshape

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# CNN + LSTM 모델 설계
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

# 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', verbose=1)
]

# 학습
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=callbacks)

# 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🔥 최종 테스트 정확도: {test_acc:.4f}")

# 라벨 인코더 저장도 추천
import joblib
joblib.dump(le, "label_encoder.pkl")
