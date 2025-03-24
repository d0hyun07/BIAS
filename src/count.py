import os

base_dir = "C:/Users/u0102/Desktop/BIAS/data/raw_audio"  # 너가 사용하는 경로로 맞춰줘

total = 0
genres = {}

for root, dirs, files in os.walk(base_dir):
    count = len([f for f in files if f.endswith('.wav')])
    if count > 0:
        genre = os.path.basename(root)
        genres[genre] = count
        total += count

print("🎸 폴더별 오디오 개수:")
for genre, count in genres.items():
    print(f" - {genre}: {count}개")

print(f"\n🎯 총 오디오 파일 수: {total}개")
