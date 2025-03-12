import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 데이터 로드
df = pd.read_csv("guitar_features.csv")

# 입력(X)과 출력(y) 분리
X = df.iloc[:, :-1].values  # 특징 데이터
y = np.array(["rock"])  # 예제 데이터 (향후 여러 장르로 확장 가능)

# 라벨 인코딩
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 데이터 정규화 (특징 스케일링)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/테스트 데이터셋 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
