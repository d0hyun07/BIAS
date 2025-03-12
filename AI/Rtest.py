import pandas as pd

df = pd.read_csv("guitar_features.csv")
print(df["label"].value_counts())  # 각 장르별 데이터 개수 출력
