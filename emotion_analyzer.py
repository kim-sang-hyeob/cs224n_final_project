#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoBERT Emotion Analyzer
txt 파일의 시를 문장 단위로 감정 분석
"""
import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ──────────────────────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
DEFAULT_KOBERT_DIR = PROJ_ROOT / "KoBERT" / "kobert-emotion"
POEM_DIR = PROJ_ROOT / "generated_poems"
EMOTION_OUTPUT_DIR = PROJ_ROOT / "emotion_analysis"

def load_kobert_local(ckpt_dir: str, device: str = "cpu"):
    """학습된 KoBERT 모델 로드"""
    try:
        ckpt_path = Path(ckpt_dir)
        
        # 최신 체크포인트 찾기
        if not (ckpt_path / "config.json").exists():
            checkpoint_dirs = [d for d in ckpt_path.iterdir() 
                             if d.is_dir() and d.name.startswith("checkpoint-")]
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, 
                                      key=lambda x: int(x.name.split("-")[-1]))
                ckpt_dir = str(latest_checkpoint)
                print(f"[INFO] Using latest checkpoint: {ckpt_dir}")
        
        print(f"[INFO] Loading your trained KoBERT from: {ckpt_dir}")
        
        # 토크나이저 로드
        tok = AutoTokenizer.from_pretrained(
            ckpt_dir,
            use_fast=False,
            trust_remote_code=True
        )
        
        # 모델 로드 (7개 감정 클래스)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir,
            num_labels=7,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        ).to(device)
        mdl.eval()
        
        print("[INFO] KoBERT loaded successfully!")
        print("[INFO] Emotion classes: 분노, 슬픔, 불안, 당황, 상처, 무감정, 기쁨")
        return tok, mdl
        
    except Exception as e:
        print(f"[ERROR] Failed to load KoBERT: {e}")
        sys.exit(1)

def predict_emotion(tok, mdl, text: str) -> Tuple[str, float]:
    """감정 예측"""
    if not text.strip():
        return "무감정", 0.0
    
    # 텍스트 인코딩
    enc = tok(text, return_tensors="pt", truncation=True, 
              padding=True, max_length=128)
    
    # GPU 사용시 디바이스 맞춤
    if next(mdl.parameters()).is_cuda:
        enc = {k: v.cuda() for k, v in enc.items()}
    
    with torch.no_grad():
        logits = mdl(**enc).logits
    
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    
    # 7개 감정 라벨
    id2lab = {0:"분노", 1:"슬픔", 2:"불안", 3:"당황", 4:"상처", 5:"무감정", 6:"기쁨"}
    
    predicted_id = int(probs.argmax())
    confidence = float(probs.max())
    
    return id2lab[predicted_id], confidence

def split_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 분할"""
    # 한국어 문장 분할 (마침표, 물음표, 느낌표 기준)
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def analyze_poem_emotions(poem_file: str, kobert_dir: str) -> List[Tuple[str, str, float]]:
    """시 파일의 감정 분석"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # KoBERT 로드
    tok, mdl = load_kobert_local(kobert_dir, device="cpu")  # 안정성을 위해 CPU 사용
    
    # 시 파일 읽기
    with open(poem_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 실제 시 내용만 추출 (메타데이터 제외)
    lines = content.split('\n')
    poem_start = False
    poem_lines = []
    
    for line in lines:
        if line.strip() == "=" * 50:
            if poem_start:
                break
            poem_start = True
            continue
        if poem_start and line.strip():
            poem_lines.append(line.strip())
    
    poem_text = '\n'.join(poem_lines)
    
    # 문장 단위로 분할
    sentences = split_sentences(poem_text)
    
    # 각 문장 감정 분석
    results = []
    print(f"\n[Emotion Analysis] Processing {len(sentences)} sentences")
    print("-" * 60)
    
    for i, sentence in enumerate(sentences, 1):
        if len(sentence) < 3:  # 너무 짧은 문장 스킵
            continue
            
        emotion, confidence = predict_emotion(tok, mdl, sentence)
        results.append((sentence, emotion, confidence))
        
        print(f"{i:2d}. {sentence}")
        print(f"    → {emotion} (신뢰도: {confidence:.3f})")
        print()
    
    return results

def save_emotion_analysis(results: List[Tuple[str, str, float]], 
                         original_file: str, output_dir: Path) -> str:
    """감정 분석 결과 저장"""
    output_dir.mkdir(exist_ok=True)
    
    # 출력 파일명 생성
    original_name = Path(original_file).stem
    output_filename = f"emotion_{original_name}.txt"
    output_path = output_dir / output_filename
    
    # 감정 통계 계산
    emotion_counts = {}
    for _, emotion, _ in results:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Emotion Analysis Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Original file: {original_file}\n")
        f.write(f"Total sentences: {len(results)}\n\n")
        
        f.write("Emotion Distribution:\n")
        for emotion, count in emotion_counts.items():
            f.write(f"  {emotion}: {count}개 ({count/len(results)*100:.1f}%)\n")
        f.write("\n" + "=" * 60 + "\n\n")
        
        f.write("Detailed Analysis:\n")
        f.write("-" * 60 + "\n")
        for i, (sentence, emotion, confidence) in enumerate(results, 1):
            f.write(f"{i:2d}. {sentence}\n")
            f.write(f"    → {emotion} (신뢰도: {confidence:.3f})\n\n")
    
    print(f"[INFO] Analysis saved to: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KoBERT Emotion Analyzer")
    parser.add_argument("--poem_file", help="시 파일 경로")
    parser.add_argument("--kobert_dir", default=str(DEFAULT_KOBERT_DIR))
    parser.add_argument("--poem_dir", default=str(POEM_DIR), 
                       help="시 파일들이 있는 디렉토리 (최신 파일 자동 선택)")
    args = parser.parse_args()
    
    # 파일 선택
    if args.poem_file:
        poem_file = args.poem_file
    else:
        # 가장 최근 생성된 시 파일 자동 선택
        poem_files = list(Path(args.poem_dir).glob("poem_*.txt"))
        if not poem_files:
            print("[ERROR] No poem files found in", args.poem_dir)
            sys.exit(1)
        poem_file = str(max(poem_files, key=lambda x: x.stat().st_mtime))
        print(f"[INFO] Using latest poem file: {poem_file}")
    
    # 감정 분석 실행
    results = analyze_poem_emotions(poem_file, args.kobert_dir)
    
    # 결과 저장
    output_file = save_emotion_analysis(results, poem_file, EMOTION_OUTPUT_DIR)
    
    print(f"\n[Complete] Emotion analysis saved to: {output_file}")
