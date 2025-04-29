from pydub import AudioSegment
import os

def split_audio_every_4s(file_path, save_dir, clip_length_sec=2, step_sec=4):
    audio = AudioSegment.from_file(file_path)
    total_length_ms = len(audio)
    clip_length_ms = clip_length_sec * 1000
    step_ms = step_sec * 1000

    os.makedirs(save_dir, exist_ok=True)

    count = 135
    for start in range(0, total_length_ms, step_ms):
        end = start + clip_length_ms
        if end > total_length_ms:
            break  # 끝부분 잘린 건 버리기
        clip = audio[start:end]
        if len(clip) >= clip_length_ms * 0.9:  # 90% 이상이어야 저장
            filename = os.path.join(save_dir, f"{count:03d}.wav")
            clip.export(filename, format="wav")
            print(f"Saved {filename}")
            count += 1

# 사용 예시
split_audio_every_4s(
    file_path="C:/Users/u0102/Desktop/BIAS/data/data2/Gm/Gm_dist.wav",  # 긴 녹음 파일
    save_dir="C:/Users/u0102/Desktop/BIAS/data/data2/Gm",             # 저장할 폴더
    clip_length_sec=2,                                  # 2초짜리로 자르고
    step_sec=4                                          # 4초 간격으로 시작
)
