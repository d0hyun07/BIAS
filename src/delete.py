import os

base_dir = "C:/Users/u0102/Downloads/audio_mono-pickup_mix"

deleted_files = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.wav') and ('solo' in file.lower() or 'lead' in file.lower()):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            deleted_files.append(file_path)

print(f"🗑️ 삭제된 파일 수: {len(deleted_files)}")
for f in deleted_files:
    print(f"삭제: {f}")
