import os

base_dir = "data/data2"

for chord in os.listdir(base_dir):
    chord_path = os.path.join(base_dir, chord)
    if not os.path.isdir(chord_path):
        continue

    wav_files = sorted([f for f in os.listdir(chord_path) if f.endswith(".wav")])

    for i, filename in enumerate(wav_files):
        new_name = f"{chord}_{i+1:03d}.wav"
        src = os.path.join(chord_path, filename)
        dst = os.path.join(chord_path, new_name)

        os.rename(src, dst)
        print(f"✅ {filename} → {new_name}")

print("🎉 파일 이름 정리 완료")
