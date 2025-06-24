import os
import torch
import numpy as np

# ------------------------------------------------------------
# 1) GPU 대신 CPU만 사용하도록 강제 (CUDA 인덱싱 오류 디버깅용)
# ------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# 2) 모델 디렉터리 & 레이블 매핑 
# ------------------------------------------------------------
MODEL_DIR = "cs224n_default_proj\KoBERT\kobert-emotion"
id2label = {
    0: "분노",
    1: "슬픔",
    2: "불안",
    3: "당황",
    4: "상처",
    5: "무감정",
    6: "기쁨",
}

# ------------------------------------------------------------
# 3) CPU 디바이스 설정
# ------------------------------------------------------------
device = torch.device("cpu")

# ------------------------------------------------------------
# 4) 토크나이저 & 모델 로드
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ------------------------------------------------------------
# 5) 예측 함수 정의 (인덱스 범위 체크 포함, 예외 처리)
# ------------------------------------------------------------
def predict_emotion(text: str):
    """
    입력 문장(text)에 대해 감정 레이블과 각 레이블 확률 리스트를 반환합니다.
    토큰화 후, 입력 아이디들이 모델의 어휘 크기(vocab_size) 범위 내에 있는지 확인합니다.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("입력 문장은 비어 있을 수 없습니다.")

    # 토큰화: 배치 차원 추가, padding=True로 짧은 문장도 처리
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    # CPU 모드
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # ------------------------------------------------------------
    # ----- 인덱스 범위 디버깅: 어휘(vocab) 크기보다 큰 인덱스 확인 -----
    # ------------------------------------------------------------
    vocab_size = tokenizer.vocab_size  # or model.config.vocab_size
    max_id = int(input_ids.max().item())
    if max_id >= vocab_size:
        raise IndexError(
            f"토큰화된 입력에 어휘 크기(vocab_size={vocab_size})를 초과하는 인덱스가 있습니다: max_id={max_id}"
        )

    # ------------------------------------------------------------
    # ----- 모델 추론 -----
    # ------------------------------------------------------------
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: (1, num_labels)

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()  # shape: (num_labels,)

    # 확률이 높은 순서로 정렬
    sorted_indices = np.argsort(probs)[::-1]
    scores = [(id2label[i], float(probs[i])) for i in sorted_indices]

    pred_label = id2label[sorted_indices[0]]
    return pred_label, scores

def safe_predict_emotion(text: str):
    """
    predict_emotion 호출 시 예외가 발생하면, None 반환
    """
    try:
        return predict_emotion(text)
    except Exception as e:
        print(f"[오류] 예측 중 예외 발생: {e}")
        return None, None

# ------------------------------------------------------------
# 6) CLI 루프: 사용자 입력을 받아 실시간 예측
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== KoBERT 감정 분류기 (CPU 모드) ===")
    print(f"모델 경로: {MODEL_DIR}")
    print("종료하려면 빈 줄 입력 후 Enter\n")

    while True:
        text = input("문장 입력 > ").strip()
        if text == "":
            print("종료합니다.")
            break

        label, scores = safe_predict_emotion(text)
        if label is None:
            # 예측 실패 케이스: 사용자에게 알림 후 다음 입력 대기
            continue

        print(f"▶ 예측 감정: {label}")
        print("  • 상위 5개 확률:")
        for lab, prob in scores[:5]:
            print(f"    - {lab}: {prob:.3f}")
        print()
