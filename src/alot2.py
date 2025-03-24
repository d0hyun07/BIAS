import os
import librosa
import numpy as np
import soundfile as sf
import scipy.signal

base_dir = "C:/Users/u0102/Desktop/BIAS/data/raw_audio"
genres = ['funk', 'jazz', 'rock', 'pop']
tempos = ['slow', 'middle', 'fast']

def add_light_noise(y, noise_level=0.002):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def light_eq(y):
    y_fft = np.fft.rfft(y)
    y_fft[int(len(y_fft)*0.6):int(len(y_fft)*0.8)] *= 1.1
    return np.fft.irfft(y_fft)

def light_reverb(y):
    reverb_kernel = np.zeros(1500)
    reverb_kernel[0] = 1
    reverb_kernel[1000] = 0.4
    return np.convolve(y, reverb_kernel, mode='same')

for genre in genres:
    for tempo in tempos:
        folder = os.path.join(base_dir, genre, tempo)
        for file in os.listdir(folder):
            if "noise" in file or "eq" in file or "reverb" in file:  
                continue  # 이미 증강된 파일은 패스

            file_path = os.path.join(folder, file)
            y, sr = librosa.load(file_path, sr=22050)
            base_name = os.path.splitext(file)[0]

            # 증강 파일 이름 규칙
            sf.write(os.path.join(folder, f"{base_name}_noise.wav"), add_light_noise(y), sr)
            sf.write(os.path.join(folder, f"{base_name}_reverb.wav"), light_reverb(y), sr)
            sf.write(os.path.join(folder, f"{base_name}_eq.wav"), light_eq(y), sr)

            print(f"✅ {genre}/{tempo} 폴더 - {file} → 증강 완료")

print("🎉 최종 증강 완료! _noise, _eq, _reverb 네이밍으로 저장됨")
