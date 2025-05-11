def analyze_progression(wav_path, model, label_encoder, window_size=2.0, hop_size=1.0, sr=44100,
                        n_mels=128, n_fft=1024, hop_length=512, device='cpu'):
    import librosa
    import numpy as np
    import torch

    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    model.eval()
    y, _ = librosa.load(wav_path, sr=sr)
    total_duration = librosa.get_duration(y=y, sr=sr)

    predictions = []

    for start in np.arange(0, total_duration - window_size + 0.01, hop_size):
        end = start + window_size
        y_slice = y[int(start * sr):int(end * sr)]

        if len(y_slice) < int(window_size * sr):
            y_slice = np.pad(y_slice, (0, int(window_size * sr) - len(y_slice)))

        mel = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_fft=n_fft,
                                             hop_length=hop_length, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel_db = standardize(mel_db)

        input_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_tensor)
            pred_idx = torch.argmax(out, dim=1).item()
            chord = label_encoder.inverse_transform([pred_idx])[0]
            predictions.append(chord)

    # 중복 제거
    result = [predictions[0]] if predictions else []
    for chord in predictions[1:]:
        if chord != result[-1]:
            result.append(chord)

    return result
