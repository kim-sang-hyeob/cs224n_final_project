import os
import json
import numpy as np
import torch

from datasets import Dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

# -------------------------- 설정 --------------------------
# 1단계에서 Fine-tuning된 체크포인트 폴더 경로
# (예: "kobert-emotion/checkpoint-27280" 등, 실제 경로에 맞게 수정하세요)
BASE_CHECKPOINT = "kobert-emotion/checkpoint-27280"

# Full Fine-Tuning 후 저장할 폴더
OUTPUT_DIR      = "./kobert-emotion-fullft-emotionjson"

# 추가 학습용 로컬 JSON 파일 경로
HQ_TRAIN_JSON   = "emotion_train_dataset.json"
HQ_VALID_JSON   = "emotion_valid_dataset.json"

# 하이퍼파라미터
MODEL_NAME   = "skt/kobert-base-v1"
MAX_LEN      = 128
BATCH_SIZE   = 8        # Full FT 시 메모리 이슈가 있을 수 있으므로 8 정도로 시작
LR           = 5e-6     # 아주 작은 학습률 (5e-6 ~ 1e-6 권장)
EPOCHS       = 7        # 2~3 에포크 정도로 시작, 과적합 주의
EARLY_STOPPING_PATIENCE = 1  # 검증 F1이 1 에포크 동안 개선되지 않으면 중단
SEED         = 42

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- 1) 레이블 매핑 (1단계와 동일) ----------------------
# 반드시 1단계와 같은 순서를 사용해야 합니다.
id2label = {
    0: "분노",
    1: "슬픔",
    2: "불안",
    3: "당황",
    4: "상처",
    5: "무감정",
    6: "기쁨",
}
label2id = {v: k for k, v in id2label.items()}

# -------------------- 2) 고품질 데이터 로드 & Dataset 생성 ----------------------
# emotion_train_dataset.json 과 emotion_valid_dataset.json 파일은 다음과 같은 형식이어야 합니다:
# [
#   { "content": "문장1...", "emotion": "슬픔" },
#   { "content": "문장2...", "emotion": "기쁨" },
#   ...
# ]

with open(HQ_TRAIN_JSON, encoding="utf-8") as f:
    train_raw = json.load(f)

with open(HQ_VALID_JSON, encoding="utf-8") as f:
    valid_raw = json.load(f)

def convert_example(ex):
    return {
        "text":  ex["content"],
        "label": label2id[ex["emotion"]]
    }

train_list = [convert_example(x) for x in train_raw]
valid_list = [convert_example(x) for x in valid_raw]

train_ds = Dataset.from_list(train_list)
valid_ds = Dataset.from_list(valid_list)

# -------------------- 3) 토크나이저 & 토큰화 ----------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def tokenize_fn(batch):
    enc = tok(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,            # DataCollatorWithPadding가 패딩 처리
        return_token_type_ids=False
    )
    enc.pop("token_type_ids", None)
    return enc

# "text" 컬럼을 없애고, 토큰화된 결과(input_ids, attention_mask)만 남김
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
valid_ds = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# 필요한 컬럼(input_ids, attention_mask, label)만 tensor로 변환
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

# -------------------- 4) 1단계 체크포인트 로드 (Full Fine-Tuning) ----------------------------
# 1단계 학습된 체크포인트에서 모델을 불러온 뒤, 전체 파라미터를 학습 가능하게 설정
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_CHECKPOINT,
    num_labels   = len(id2label),  # 7개 클래스
    id2label     = id2label,
    label2id     = label2id,
)

# 모든 파라미터를 학습 가능하게 설정 (requires_grad=True)
for param in base_model.parameters():
    param.requires_grad = True

# -------------------- 5) 평가 지표 정의 ----------------------------
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# -------------------- 6) TrainingArguments & Trainer -------------------
training_args = TrainingArguments(
    output_dir               = OUTPUT_DIR,
    evaluation_strategy      = "epoch",
    save_strategy            = "epoch",
    load_best_model_at_end   = True,
    metric_for_best_model    = "macro_f1",
    greater_is_better        = True,

    num_train_epochs         = EPOCHS,
    per_device_train_batch_size  = BATCH_SIZE,
    per_device_eval_batch_size   = BATCH_SIZE,
    learning_rate            = LR,
    weight_decay             = 0.01,
    fp16                     = torch.cuda.is_available(),

    report_to                = "none",
    seed                     = SEED,
    save_total_limit         = 2,
)

trainer = Trainer(
    model           = base_model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = valid_ds,
    tokenizer       = tok,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

# -------------------- 7) 학습 및 저장 --------------------------
if __name__ == "__main__":
    print(f"▶️ 고품질 학습 샘플 수: {len(train_ds)}")
    print(f"▶️ 고품질 검증 샘플 수: {len(valid_ds)}")

    print("\n🚀 Full Fine-Tuning (추가 학습) 시작...\n")
    trainer.train()

    print("\n💾 모델 저장 중...")
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    saved_files = os.listdir(OUTPUT_DIR)
    print(f"\n✅ 저장 완료! '{OUTPUT_DIR}' 폴더 내 파일 목록:")
    for fn in sorted(saved_files):
        print("   •", fn)
    print(f"📁 모델 경로: {os.path.abspath(OUTPUT_DIR)}")
