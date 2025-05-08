import os

AUG_DIR = "augmented_data"  # 증강된 전체 루트 폴더

for chord in os.listdir(AUG_DIR):
    chord_path = os.path.join(AUG_DIR, chord)
    if not os.path.isdir(chord_path):
        continue

    for filename in os.listdir(chord_path):
        if not filename.endswith(".wav"):
            continue

        # 남길 조건: 파일명에 아래 중 하나 포함
        if any(x in filename for x in ["_orig", "_volup", "_voldown"]):
            continue

        # 삭제
        file_path = os.path.join(chord_path, filename)
        os.remove(file_path)
        print(f"🗑️ deleted: {file_path}")

print("✅ 불필요한 증강 파일 삭제 완료!")
