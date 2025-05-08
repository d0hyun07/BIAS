import os

ROOT_DIR = "C:/Users/u0102/Desktop/BIAS/data/augmented_data"  # 대상 폴더
total = 0

print("📊 코드별 WAV 파일 개수:\n")

for chord in sorted(os.listdir(ROOT_DIR)):
    chord_path = os.path.join(ROOT_DIR, chord)
    if not os.path.isdir(chord_path):
        continue

    wav_files = [f for f in os.listdir(chord_path) if f.endswith(".wav")]
    count = len(wav_files)
    total += count

    print(f"{chord:<5} → {count:>4}개")

print("\n🧮 총 WAV 파일 수:", total)
