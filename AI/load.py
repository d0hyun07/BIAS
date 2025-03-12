import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 데이터 로드
df = pd.read_csv("guitar_features.csv")

# 입력(X)과 출력(y) 분리
X = df.iloc[:, :-1].values  # 특징 데이터
y = df.iloc[:, -1].values   # 장르 (레이블)

# 라벨 인코딩 (문자열 → 숫자 변환)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 데이터 정규화 (스케일링)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 데이터 형태 변환 (CNN + LSTM 입력 맞추기)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 데이터 크기 확인
print(f"훈련 데이터 크기: {X_train.shape}, 테스트 데이터 크기: {X_test.shape}")
