import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # 설정
    ROOT = "C:/Users/u0102/Desktop/BIAS/data/augmented_data"  # 로컬 경로로 변경
    CSV = os.path.join(ROOT, "metadata.csv")
    SAMPLE_RATE = 44100
    DURATION = 2
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 1024
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 클래스
    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    class ChordDataset(Dataset):
        def __init__(self, df):
            self.df = df

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = os.path.join(ROOT, row["filename"])
            y, _ = librosa.load(path, sr=SAMPLE_RATE)
            mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
                                                 hop_length=HOP_LENGTH, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            mel_db = standardize(mel_db)
            return torch.tensor(mel_db).unsqueeze(0), torch.tensor(row["label"]).long()

    # 개선된 CNN 모델
    class ImprovedCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * (N_MELS // 8) * (173 // 8), 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    # 데이터 준비
    df = pd.read_csv(CSV)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"])
    train_loader = DataLoader(ChordDataset(train_df), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(ChordDataset(val_df), batch_size=32, num_workers=0)

    # 학습 준비
    model = ImprovedCNN(num_classes=len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # 학습 루프
    for epoch in range(30):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📦 Epoch {epoch+1:02} | Train Loss: {total_loss:.4f}")

        # 검증 정확도 출력
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_acc = correct / total * 100
        print(f"🎯 Validation Accuracy: {val_acc:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), "chord_model_improved.pth")
    print("✅ 향상된 모델 저장 완료: chord_model_improved.pth")
