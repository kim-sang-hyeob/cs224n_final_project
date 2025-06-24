# ----------------------------------
# 이렇게 추론했을때 한글이 깨져서 추가로 작업 필요.. tts 를 교체해야할 것 같다. 
# 한글 tts 가 제대로 잘 작동하지는 않는다. 
# ----------------------------------


# 1) 설치 (터미널에서 한 번만 실행)
# ----------------------------------
# pip install git+https://github.com/suno-ai/bark@main
# pip install soundfile


# 2) 코드 시작
# ----------------------------------
from bark import generate_audio, preload_models
import soundfile as sf

# 3) 모델 프리로드 (최초 1회, 모델 파일 다운로드)
preload_models()

# 4) 감정 라벨 → 스타일 매핑
EMOTION_MAPPING = {
    "공포":   "무섭고 긴장감 있는 톤으로",
    "놀람":   "놀라고 흥미진진한 목소리로",
    "분노":   "화나고 강한 톤으로",
    "슬픔":   "슬프고 차분한 목소리로",
    "중립":   "평범하고 담담한 톤으로",
    "행복":   "밝고 활기찬 톤으로",
    "혐오":   "역겹고 불쾌한 목소리로"
}

# 5) 시 원문 (BERT 분류는 이 문장 단위로 수행했다고 가정)
poem_text = """
<topic: 바다 위에 뜬 달>
아득한 옛날이 있었지요
바닷가의 한낮에도
달빛은 따사로운 햇살을 뿌리며
달빛과 바람 사이로 날아다녔지요

달빛에 비친 보름달의
눈썹에 흰 별 하나가
눈썹에 흰 별 하나가
달빛을 반짝이는데

그런 밤이면
달빛이
달빛과 파도 사이를 날리고

그런 밤이면
바다의 달빛은
마음의 첨표인 듯
<endoftext>
""".strip().splitlines()

# 6) (가정) BERT 예측 함수
def bert_predict(sentence: str) -> str:
    """
    실제로는 여러분이 학습한 BERT 모델을 불러와
    각 문장에 대해 라벨 중 하나를 반환하도록 구현하세요.
    여기서는 예시로 랜덤 선택하거나, 미리 정의한 리스트를 사용합니다.
    """
    # 예시: 순서대로 감정을 지정했다고 가정
    demo = ["행복","중립","슬픔","중립","행복",
            "슬픔","슬픔","슬픔","놀람","중립",
            "슬픔","슬픔","중립","행복"]
    # 인덱스 범위에 맞춰 순환
    return demo[sentence_idx % len(demo)]

# 7) 문장별 감정 반영 및 오디오 생성
for sentence_idx, sentence in enumerate(poem_text):
    # 빈 줄이나 메타토큰(<topic>, <endoftext>) 건너뛰기
    if not sentence or sentence.startswith("<"):
        continue

    # 1) BERT로 감정 예측 (가정)
    emotion_label = bert_predict(sentence)

    # 2) 스타일 토큰 매핑
    style_desc = EMOTION_MAPPING.get(emotion_label, EMOTION_MAPPING["중립"])

    # 3) Bark용 프롬프트 생성
    prompt = f"[emotion:{style_desc}]\n{sentence}"

    print(f"💬 문장: \"{sentence}\"  →  감정: {emotion_label} ({style_desc})")

    # 4) 음성 생성
    audio_array = generate_audio(prompt)

    # 5) WAV 파일로 저장
    out_fname = f"poem_line_{sentence_idx+1}_{emotion_label}.wav"
    sf.write(out_fname, audio_array, samplerate=24000)
    print(f"▶ 저장: {out_fname}\n")

print("모든 문장에 대한 TTS 생성 완료!")    