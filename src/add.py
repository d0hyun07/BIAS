import os
from pydub import AudioSegment

INPUT_ROOT = "data/data2"          # 정리된 원본 폴더
OUTPUT_ROOT = "augmented_data"     # 증강된 파일 저장 폴더
TARGET_DURATION_MS = 2000          # 고정 길이: 2초

def fix_length(audio: AudioSegment, duration_ms=2000):
    if len(audio) > duration_ms:
        return audio[:duration_ms]
    else:
        silence = AudioSegment.silent(duration=duration_ms - len(audio))
        return audio + silence

def change_volume(audio, db):
    return fix_length(audio + db, TARGET_DURATION_MS)

def change_speed(audio, speed):
    changed = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    }).set_frame_rate(audio.frame_rate)
    return fix_length(changed, TARGET_DURATION_MS)

for chord_name in os.listdir(INPUT_ROOT):
    chord_path = os.path.join(INPUT_ROOT, chord_name)
    if not os.path.isdir(chord_path):
        continue

    output_dir = os.path.join(OUTPUT_ROOT, chord_name)
    os.makedirs(output_dir, exist_ok=True)

    for file in sorted(os.listdir(chord_path)):
        if not file.endswith(".wav"):
            continue

        input_path = os.path.join(chord_path, file)
        audio = AudioSegment.from_wav(input_path)
        audio = fix_length(audio, TARGET_DURATION_MS)

        base_name = os.path.splitext(file)[0]

        # 원본 복사본 (선택)
        audio.export(f"{output_dir}/{base_name}_orig.wav", format="wav")

        # 증강: 볼륨
        change_volume(audio, 3).export(f"{output_dir}/{base_name}_volup.wav", format="wav")
        change_volume(audio, -3).export(f"{output_dir}/{base_name}_voldown.wav", format="wav")

        # 증강: 속도
        change_speed(audio, 1.05).export(f"{output_dir}/{base_name}_faster.wav", format="wav")
        change_speed(audio, 0.95).export(f"{output_dir}/{base_name}_slower.wav", format="wav")

    print(f"✅ {chord_name} 코드 증강 완료")

print("🎉 전체 코드 폴더 증강 완료!")
