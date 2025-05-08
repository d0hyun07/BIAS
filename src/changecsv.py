import os
import csv

ROOT_DIR = "C:/Users/u0102/Desktop/BIAS/data/augmented_data"
CSV_PATH = os.path.join(ROOT_DIR, "metadata.csv")

rows = []

for chord in sorted(os.listdir(ROOT_DIR)):
    chord_path = os.path.join(ROOT_DIR, chord)
    if not os.path.isdir(chord_path):
        continue

    for file in sorted(os.listdir(chord_path)):
        if file.endswith(".wav"):
            filepath = f"{chord}/{file}"
            rows.append([filepath, chord])

# CSV 저장
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print(f"✅ metadata.csv 생성 완료! 총 {len(rows)}개 항목")
