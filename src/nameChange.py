import os
import shutil
import re

base_dir = "C:/Users/u0102/Downloads/audio_mono-pickup_mix"
genres_map = {
    'Jazz': 'jazz',
    'Funk': 'funk',
    'BN': 'pop',
    'Rock': 'rock',
    'Metal': 'metal',
    'SS': 'pop'  # SS는 pop으로 분류
}

tempo_ranges = {
    'slow': range(60, 91),
    'middle': range(91, 131),
    'fast': range(131, 201)
}

files = [f for f in os.listdir(base_dir) if f.endswith('.wav')]

counter = {}

for file in files:
    file_path = os.path.join(base_dir, file)

    # 장르 자동 인식 (파일명에서 앞부분 인식)
    genre_match = re.search(r'_(Jazz|Funk|BN|Rock|Metal|SS)', file)
    if genre_match:
        genre_key = genre_match.group(1)
        genre = genres_map[genre_key]
    else:
        print(f"❌ 장르 못 찾음: {file}")
        continue

    # BPM 추출
    bpm_match = re.search(r'-(\d{2,3})-[A-Gb#]', file)
    bpm = int(bpm_match.group(1)) if bpm_match else 120

    # 템포 분류
    if 60 <= bpm <= 90:
        tempo_label = 'slow'
    elif 91 <= bpm <= 130:
        tempo_label = 'middle'
    else:
        tempo_label = 'fast'

    # 폴더 생성
    dest_folder = os.path.join(base_dir, genre, tempo_label)
    os.makedirs(dest_folder, exist_ok=True)

    # 카운터 생성 or 증가
    counter.setdefault((genre, tempo_label), 1)
    count = counter[(genre, tempo_label)]

    # 리네임 후 이동
    new_name = f"{genre}_{tempo_label}_{count}.wav"
    new_path = os.path.join(dest_folder, new_name)
    shutil.move(file_path, new_path)
    print(f"✅ {file} → {genre}/{tempo_label}/{new_name}")

    counter[(genre, tempo_label)] += 1

print("🎉 SS → pop 포함 폴더 자동 생성 + 이동 완료!")
