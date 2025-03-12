import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 샘플 기타 오디오 파일 불러오기
file_path = librosa.example("nutcracker")  # 샘플 오디오 사용 (실제 기타 데이터로 변경 가능)
y, sr = librosa.load(file_path, sr=22050)

# 오디오의 스펙트로그램 분석
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of Guitar Audio")
plt.show()
