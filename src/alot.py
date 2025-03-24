import os
import librosa
import numpy as np
import soundfile as sf
import scipy.signal

base_dir = "C:/Users/u0102/Desktop/BIAS/data/raw_audio"
genres = ['funk', 'jazz', 'rock', 'pop']
tempos = ['slow', 'middle', 'fast']

def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def custom_pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)



def eq_high(y):
    y_fft = np.fft.rfft(y)
    y_fft[int(len(y_fft)*0.75):] *= 1.5
    return np.fft.irfft(y_fft)

def bandpass_filter(y, sr):
    b, a = scipy.signal.butter(4, [500/(sr/2), 3000/(sr/2)], btype='band')
    return scipy.signal.lfilter(b, a, y)

def lowpass_filter(y, sr):
    b, a = scipy.signal.butter(4, 3000/(sr/2), btype='low')
    return scipy.signal.lfilter(b, a, y)

def soft_clip(y, threshold=0.5):
    return np.tanh(y / threshold)

def volume_shift(y, gain=1.2):
    return y * gain

def reverb(y):
    reverb_kernel = np.zeros(2000)
    reverb_kernel[0] = 1
    reverb_kernel[1000] = 0.6
    reverb_kernel[1500] = 0.3
    return np.convolve(y, reverb_kernel, mode='same')

for genre in genres:
    for tempo in tempos:
        folder = os.path.join(base_dir, genre, tempo)
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                file_path = os.path.join(folder, file)
                y, sr = librosa.load(file_path, sr=22050)
                base_name = os.path.splitext(file)[0]

                # 증강 저장
                sf.write(os.path.join(folder, f"{base_name}_noise.wav"), add_noise(y), sr)
                sf.write(os.path.join(folder, f"{base_name}_pitchup.wav"), custom_pitch_shift(y, sr, 2), sr)
                sf.write(os.path.join(folder, f"{base_name}_pitchdown.wav"), custom_pitch_shift(y, sr, -2), sr)
                sf.write(os.path.join(folder, f"{base_name}_highboost.wav"), eq_high(y), sr)
                sf.write(os.path.join(folder, f"{base_name}_reverse.wav"), y[::-1], sr)
                sf.write(os.path.join(folder, f"{base_name}_bandpass.wav"), bandpass_filter(y, sr), sr)
                sf.write(os.path.join(folder, f"{base_name}_lowpass.wav"), lowpass_filter(y, sr), sr)
                sf.write(os.path.join(folder, f"{base_name}_clip.wav"), soft_clip(y), sr)
                sf.write(os.path.join(folder, f"{base_name}_volup.wav"), volume_shift(y, 1.3), sr)
                sf.write(os.path.join(folder, f"{base_name}_voldown.wav"), volume_shift(y, 0.7), sr)
                sf.write(os.path.join(folder, f"{base_name}_reverb.wav"), reverb(y), sr)

                print(f"✅ {genre}/{tempo} 폴더 - {file} → 11배 증강 완료")

print("🎉 🚀 대량 증강 완료! 1개 → 11개로 폭발!")
