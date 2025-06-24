#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP 기반 한-영 매핑 이미지 분류기 (HuggingFace 버전)
-----------------------------------------------------
1) topics.txt  ─ 한국어 토픽 목록(한 줄에 하나)
2) korean_to_english  ─ 직접 정의한 한→영 매핑 사전
3) HuggingFace CLIP 모델 로드 & 영어 prompt 임베딩 캐싱
4) classify(img_path)  ─ 이미지 하나를 (ko, en, score) 반환
5) main()              ─ CLI: 폴더/파일 스트림 모두 지원
"""

import argparse, os, sys, json, time, torch
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# ──────────────────────────────────────────────────────────────
# 1. 토픽 목록 & 한→영 매핑 (전체 매핑 추가)
# ──────────────────────────────────────────────────────────────
def load_topics(topics_file: str) -> List[str]:
    """topics.txt 파일에서 한국어 토픽 목록 로드"""
    try:
        with open(topics_file, encoding="utf-8") as f:
            topics = [line.strip().strip("'- •") for line in f if line.strip()]
        return topics
    except FileNotFoundError:
        print(f"[ERROR] {topics_file} 파일을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

# 전체 토픽에 대한 한→영 매핑 (topics.txt 기준으로 완성)
korean_to_english = {
    "사물": "object",        "꽃": "flower",            "자연": "nature",
    "일상": "everyday life", "문학": "literature",      "음식": "food",
    "사람": "person",        "삶": "life",              "성소": "sanctuary",
    "나무": "tree",          "새": "bird",              "관계": "relationship",
    "고뇌": "anguish",       "길": "road",              "물고기": "fish",
    "마을": "village",       "사랑": "love",            "식물": "plant",
    "몸": "body",            "동물": "animal",          "고립": "isolation",
    "가족": "family",        "종교": "religion",        "곤충": "insect",
    "세월": "passage of time","상실": "loss",           "갈등": "conflict",
    "기억": "memory",        "죽음": "death",           "슬픔": "sadness",
    "소통": "communication", "희망": "hope",            "도시": "city",
    "비": "rain",            "눈": "snow",              "외출": "outing",
    "옷": "clothes",         "봄": "spring",            "겨울": "winter",
    "기쁨": "joy",           "가을": "autumn",          "여름": "summer",
    "교통수단": "transportation",
    "달": "moon",            "밤": "night",             "물결": "wave",
    "의례": "ritual",        "역사": "history",         "명절": "festival",
    "상태": "state",         "빛": "light",             "해": "sun",
    "저녁": "evening",       "병": "illness",           "별": "star",
    "바람": "wind",          "날씨": "weather",         "새벽": "dawn",
    "안개": "fog",           "하늘": "sky"
}

# ──────────────────────────────────────────────────────────────
# 2. HuggingFace CLIP 모델 
# ──────────────────────────────────────────────────────────────
class ClipKoreanClassifier:
    def __init__(self,
                 topics_file: str = "topics.txt",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 model_name: str = "openai/clip-vit-base-patch32"):
        self.device = device
        print(f"[INFO] HuggingFace CLIP 모델 로딩 중... (device: {device})")
        
        # HuggingFace CLIP 모델 및 프로세서 로드
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 토픽 로드 및 매핑 확인
        all_korean_topics = load_topics(topics_file)
        self.korean_labels = []
        self.english_labels = []
        
        missing_topics = []
        for ko in all_korean_topics:
            en = korean_to_english.get(ko)
            if en is None:
                missing_topics.append(ko)
                continue
            self.korean_labels.append(ko)
            self.english_labels.append(en)
        
        if missing_topics:
            print(f"[WARNING] 매핑 누락된 토픽들: {missing_topics}", file=sys.stderr)
        
        print(f"[INFO] 총 {len(self.korean_labels)}개 토픽 로드됨")
        
        # 텍스트 임베딩 생성 및 캐싱
        self._encode_text_prompts()

    def _encode_text_prompts(self):
        """텍스트 프롬프트를 인코딩하고 정규화"""
        prompts = [f"a photo of {label}" for label in self.english_labels]
        
        # HuggingFace processor로 텍스트 처리
        text_inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            # 텍스트 임베딩 추출
            text_outputs = self.model.get_text_features(**text_inputs)
            self.text_emb = F.normalize(text_outputs, dim=-1)

    @torch.no_grad()
    def classify(self, img_path: str, topk: int = 1) -> List[Tuple[str, str, float]]:
        """
        이미지 분류 (topk 결과 반환)
        Returns: [(korean, english, score), ...]
        """
        try:
            # 이미지 로드 및 전처리
            img = Image.open(img_path).convert('RGB')
            
            # HuggingFace processor로 이미지 처리
            image_inputs = self.processor(
                images=img,
                return_tensors="pt"
            ).to(self.device)
            
            # 이미지 임베딩 추출
            image_outputs = self.model.get_image_features(**image_inputs)
            img_emb = F.normalize(image_outputs, dim=-1)
            
            # 유사도 계산
            similarities = (img_emb @ self.text_emb.T)[0]
            
            # topk 결과 추출
            top_scores, top_indices = similarities.topk(min(topk, len(self.korean_labels)))
            
            results = []
            for i in range(len(top_indices)):
                idx = top_indices[i].item()
                score = top_scores[i].item()
                results.append((
                    self.korean_labels[idx],
                    self.english_labels[idx],
                    score
                ))
            
            return results
            
        except Exception as e:
            print(f"[ERROR] 이미지 처리 실패 ({img_path}): {e}", file=sys.stderr)
            return []

    def classify_batch(self, image_paths: List[str], topk: int = 1) -> List[List[Tuple[str, str, float]]]:
        """
        배치 단위로 이미지 분류 (성능 향상)
        """
        try:
            # 이미지들 로드
            images = []
            valid_paths = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"[WARNING] 이미지 로드 실패 ({img_path}): {e}", file=sys.stderr)
            
            if not images:
                return []
            
            # 배치 처리
            image_inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # 배치 이미지 임베딩
                image_outputs = self.model.get_image_features(**image_inputs)
                img_embs = F.normalize(image_outputs, dim=-1)
                
                # 배치 유사도 계산
                similarities = img_embs @ self.text_emb.T
                
                # 각 이미지별 topk 결과
                batch_results = []
                for i, sim in enumerate(similarities):
                    top_scores, top_indices = sim.topk(min(topk, len(self.korean_labels)))
                    
                    results = []
                    for j in range(len(top_indices)):
                        idx = top_indices[j].item()
                        score = top_scores[j].item()
                        results.append((
                            self.korean_labels[idx],
                            self.english_labels[idx],
                            score
                        ))
                    batch_results.append(results)
                
                return batch_results
                
        except Exception as e:
            print(f"[ERROR] 배치 처리 실패: {e}", file=sys.stderr)
            return []

    def save_cache(self, cache_path: str = "text_embeddings.pt"):
        """텍스트 임베딩 캐시 저장"""
        cache_data = {
            'text_embeddings': self.text_emb.cpu(),
            'korean_labels': self.korean_labels,
            'english_labels': self.english_labels,
            'model_name': self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else 'openai/clip-vit-base-patch32'
        }
        torch.save(cache_data, cache_path)
        print(f"[INFO] 캐시 저장됨: {cache_path}")

# ──────────────────────────────────────────────────────────────
# 3. 유틸리티 함수들
# ──────────────────────────────────────────────────────────────
def get_image_files(path: Path) -> List[Path]:
    """이미지 파일 목록 추출"""
    if path.is_file():
        return [path]
    elif path.is_dir():
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        files = []
        for ext in extensions:
            files.extend(path.rglob(ext))
            files.extend(path.rglob(ext.upper()))
        return sorted(files)
    else:
        return []

def format_results(results: List[Tuple[str, str, float]], show_english: bool = True) -> str:
    """결과를 보기 좋게 포맷팅"""
    if not results:
        return "분류 실패"
    
    formatted = []
    for i, (ko, en, score) in enumerate(results, 1):
        if show_english:
            formatted.append(f"{i}. {ko} ({en}) [{score:.3f}]")
        else:
            formatted.append(f"{i}. {ko} [{score:.3f}]")
    
    return " | ".join(formatted)

# ──────────────────────────────────────────────────────────────
# 4. CLI 진입점 (개선된 버전)
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace CLIP 기반 한국어 토픽 이미지 분류기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python clip_classifier.py image.jpg
  python clip_classifier.py ./images/ --topk 3
  python clip_classifier.py image.jpg --topk 5 --no-english
  python clip_classifier.py ./images/ --batch --batch-size 8
        """
    )
    parser.add_argument("path", help="이미지 파일 또는 폴더 경로")
    parser.add_argument("--topk", type=int, default=3, help="상위 k개 결과 표시 (기본값: 1)")
    parser.add_argument("--cache", action="store_true", help="텍스트 임베딩 캐시 저장")
    parser.add_argument("--no-english", action="store_true", help="영어 번역 숨기기")
    parser.add_argument("--topics", default="topics.txt", help="토픽 파일 경로")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", 
                       help="사용할 CLIP 모델 (기본값: openai/clip-vit-base-patch32)")
    parser.add_argument("--batch", action="store_true", help="배치 처리 모드")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기 (기본값: 8)")
    
    args = parser.parse_args()

    # 분류기 초기화
    try:
        clf = ClipKoreanClassifier(
            topics_file=args.topics,
            model_name=args.model
        )
    except Exception as e:
        print(f"[ERROR] 분류기 초기화 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # 캐시 저장
    if args.cache:
        clf.save_cache()

    # 이미지 파일 수집
    root_path = Path(args.path)
    image_files = get_image_files(root_path)
    
    if not image_files:
        print("처리할 이미지를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {len(image_files)}개 이미지 처리 시작...\n")

    # 배치 처리 또는 개별 처리
    if args.batch and len(image_files) > 1:
        # 배치 처리
        batch_size = args.batch_size
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_paths = [str(f) for f in batch_files]
            
            batch_results = clf.classify_batch(batch_paths, topk=args.topk)
            
            for j, results in enumerate(batch_results):
                if j < len(batch_files):
                    result_str = format_results(results, show_english=not args.no_english)
                    print(f"{batch_files[j].name:<30} → {result_str}")
    else:
        # 개별 처리
        for img_path in image_files:
            results = clf.classify(str(img_path), topk=args.topk)
            result_str = format_results(results, show_english=not args.no_english)
            print(f"{img_path.name:<30} → {result_str}")

if __name__ == "__main__":
    main()