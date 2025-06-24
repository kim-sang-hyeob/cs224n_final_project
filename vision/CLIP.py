'''
mapping 없이 clip 사용한 코드
'''

import argparse, sys, torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F

def load_labels(labels_file: str):
    try:
        with open(labels_file, encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels
    except FileNotFoundError:
        print(f"[ERROR] labels file not found: {labels_file}", file=sys.stderr)
        sys.exit(1)

def extract_clip_tokens(image_path: str, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
    """
    이미지에서 CLIP을 사용해 텍스트 토큰(설명)을 추출하는 함수
    
    Args:
        image_path (str): 이미지 파일 경로
        model_name (str): CLIP 모델 이름
        device (str): 사용할 디바이스 (cuda/cpu)
    
    Returns:
        str: 이미지에 대한 텍스트 설명
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # CLIP 모델 및 프로세서 로드
    print(f"[INFO] Loading CLIP model ({model_name}) on {device}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert("RGB")
    img_inputs = processor(images=img, return_tensors="pt").to(device)
    
    # 다양한 이미지 설명 템플릿들 (시 생성에 적합한 것들)
    image_descriptions = [
        "a beautiful landscape",
        "a peaceful scene", 
        "nature and tranquility",
        "emotions and feelings",
        "love and romance",
        "sadness and melancholy",
        "joy and happiness",
        "mystery and wonder",
        "poetry and art",
        "dreams and imagination",
        "seasons and time",
        "light and shadow",
        "colors and beauty",
        "memories and nostalgia",
        "hope and inspiration"
    ]
    
    # 텍스트 입력 처리
    text_inputs = processor(text=image_descriptions, return_tensors="pt", 
                           padding=True, truncation=True).to(device)
    
    # 이미지와 텍스트 임베딩 계산
    with torch.no_grad():
        img_feats = model.get_image_features(**img_inputs)
        text_feats = model.get_text_features(**text_inputs)
        
        # 정규화
        img_emb = F.normalize(img_feats, dim=-1)
        text_emb = F.normalize(text_feats, dim=-1)
        
        # 유사도 계산
        sims = (img_emb @ text_emb.T)[0]  # [num_descriptions]
        
        # 가장 유사한 상위 3개 설명 선택
        top_scores, top_idxs = sims.topk(3)
        
        # 상위 설명들을 조합하여 최종 토큰 생성
        top_descriptions = [image_descriptions[idx.item()] for idx in top_idxs]
        
        # 시 생성에 적합한 형태로 조합
        combined_description = ", ".join(top_descriptions)
        
        # 한국어 시 생성에 적합한 키워드로 변환
        korean_keywords = convert_to_korean_keywords(combined_description)
        
        return korean_keywords

def convert_to_korean_keywords(english_description: str) -> str:
    """
    영어 설명을 한국어 시 생성에 적합한 키워드로 변환
    """
    # 간단한 매핑 (실제로는 더 정교한 번역이나 매핑이 필요할 수 있음)
    keyword_mapping = {
        "beautiful landscape": "아름다운 풍경",
        "peaceful scene": "평화로운 장면",
        "nature and tranquility": "자연과 평온",
        "emotions and feelings": "감정과 느낌",
        "love and romance": "사랑과 로맨스",
        "sadness and melancholy": "슬픔과 우울",
        "joy and happiness": "기쁨과 행복",
        "mystery and wonder": "신비와 경이",
        "poetry and art": "시와 예술",
        "dreams and imagination": "꿈과 상상",
        "seasons and time": "계절과 시간",
        "light and shadow": "빛과 그림자",
        "colors and beauty": "색채와 아름다움",
        "memories and nostalgia": "추억과 향수",
        "hope and inspiration": "희망과 영감"
    }
    
    # 영어 키워드를 한국어로 변환
    korean_keywords = []
    for keyword in english_description.split(", "):
        if keyword in keyword_mapping:
            korean_keywords.append(keyword_mapping[keyword])
        else:
            # 매핑에 없는 경우 원본 사용
            korean_keywords.append(keyword)
    
    return ", ".join(korean_keywords)

class SimpleCLIPClassifier:
    def __init__(self,
                 labels_file: str = "topics.txt",
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading CLIP model ({model_name}) on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # 1) 레이블 로드 (영어)
        self.labels = load_labels(labels_file)
        # 2) "a photo of {label}" 형태로 프롬프트 생성
        prompts = [f"a photo of {lab}" for lab in self.labels]
        text_inputs = self.processor(text=prompts, return_tensors="pt",
                                     padding=True, truncation=True).to(self.device)
        # 3) 텍스트 임베딩 캐싱
        with torch.no_grad():
            text_feats = self.model.get_text_features(**text_inputs)
            self.text_emb = F.normalize(text_feats, dim=-1)

    @torch.no_grad()
    def classify(self, img_path: str, topk: int = 1):
        # 이미지 로드 및 전처리
        img = Image.open(img_path).convert("RGB")
        img_inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        # 이미지 임베딩
        img_feats = self.model.get_image_features(**img_inputs)
        img_emb = F.normalize(img_feats, dim=-1)

        # 유사도 계산 & 상위 topk 추출
        sims = (img_emb @ self.text_emb.T)[0]  # [num_labels]
        top_scores, top_idxs = sims.topk(min(topk, len(self.labels)))

        return [(self.labels[idx.item()], top_scores[i].item())
                for i, idx in enumerate(top_idxs)]

def get_image_files(path: Path):
    if path.is_file():
        return [path]
    if path.is_dir():
        exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.tiff','*.webp']
        files = []
        for ext in exts:
            files += list(path.rglob(ext)) + list(path.rglob(ext.upper()))
        return sorted(files)
    return []

def main():
    p = argparse.ArgumentParser(
        description="Simple CLIP Image Classifier (no mapping)",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("path", help="이미지 파일 또는 디렉터리")
    p.add_argument("--labels", default="topics.txt", help="영어 레이블 파일")
    p.add_argument("--topk", type=int, default=1, help="상위 k개 출력")
    p.add_argument("--model", default="openai/clip-vit-base-patch32",
                   help="CLIP 모델 이름")
    args = p.parse_args()

    clf = SimpleCLIPClassifier(
        labels_file=args.labels,
        model_name=args.model
    )

    files = get_image_files(Path(args.path))
    if not files:
        print(f"[ERROR] No images found in {args.path}", file=sys.stderr)
        sys.exit(1)

    for img_path in files:
        results = clf.classify(str(img_path), topk=args.topk)
        result_str = " | ".join(f"{lab} [{score:.3f}]" for lab, score in results)
        print(f"{img_path.name:<30} → {result_str}")

if __name__ == "__main__":
    main()
