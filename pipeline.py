#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision → Poetry Generator
이미지에서 시를 생성하고 txt 파일로 저장
"""
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    PreTrainedTokenizerFast, GPT2LMHeadModel
)

# ──────────────────────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = PROJ_ROOT / "img.jpg"
DEFAULT_KOGPT2_PATH = PROJ_ROOT / "best_models" / "freeze-epoch5"
DEFAULT_LABELS_FILE = PROJ_ROOT / "topics.txt"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
OUTPUT_DIR = PROJ_ROOT / "generated_poems"

# 한→영 매핑
korean_to_english = {
    "사물":"object","꽃":"flower","자연":"nature","일상":"everyday life","문학":"literature",
    "음식":"food","사람":"person","삶":"life","성소":"sanctuary","나무":"tree","새":"bird",
    "관계":"relationship","고뇌":"anguish","길":"road","물고기":"fish","마을":"village",
    "사랑":"love","식물":"plant","몸":"body","동물":"animal","고립":"isolation","가족":"family",
    "종교":"religion","곤충":"insect","세월":"passage of time","상실":"loss","갈등":"conflict",
    "기억":"memory","죽음":"death","슬픔":"sadness","소통":"communication","희망":"hope",
    "도시":"city","비":"rain","눈":"snow","외출":"outing","옷":"clothes","봄":"spring",
    "겨울":"winter","기쁨":"joy","가을":"autumn","여름":"summer","교통수단":"transportation",
    "달":"moon","밤":"night","물결":"wave","의례":"ritual","역사":"history","명절":"festival",
    "상태":"state","빛":"light","해":"sun","저녁":"evening","병":"illness","별":"star",
    "바람":"wind","날씨":"weather","새벽":"dawn","안개":"fog","하늘":"sky"
}

def load_labels(file_path: str) -> List[str]:
    """라벨 파일 로드"""
    try:
        with open(file_path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        print(f"[ERROR] labels file not found: {file_path}")
        sys.exit(1)

class ClipKoreanClassifier:
    """CLIP 한국어 분류기"""
    def __init__(self, topics_file: str, model_name: str = DEFAULT_CLIP_MODEL, 
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading CLIP ({model_name}) on {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model_name)
        
        # 토픽 로드 & 매핑
        ko_topics = load_labels(topics_file)
        self.ko_labels, self.en_labels = [], []
        for ko in ko_topics:
            en = korean_to_english.get(ko)
            if en:
                self.ko_labels.append(ko)
                self.en_labels.append(en)
        
        print(f"[INFO] {len(self.ko_labels)} labels loaded")
        
        # 텍스트 임베딩 미리 계산
        prompts = [f"a photo of {en}" for en in self.en_labels]
        text_in = self.proc(text=prompts, return_tensors="pt",
                           padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            self.text_emb = F.normalize(
                self.model.get_text_features(**text_in), dim=-1)
    
    @torch.no_grad()
    def classify(self, img_path: str, topk: int = 3) -> List[Tuple[str, float]]:
        """이미지 분류"""
        img = Image.open(img_path).convert("RGB")
        img_enc = self.proc(images=img, return_tensors="pt").to(self.device)
        img_emb = F.normalize(
            self.model.get_image_features(**img_enc), dim=-1)
        sims = (img_emb @ self.text_emb.T)[0]
        score, idx = sims.topk(min(topk, len(self.ko_labels)))
        return [(self.ko_labels[i.item()], score[j].item())
                for j, i in enumerate(idx)]

def load_kogpt2(model_dir: str, device):
    """KoGPT-2 모델 로드"""
    print(f"[INFO] Loading KoGPT-2 from {model_dir}")
    
    tok = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
        use_fast=False
    )
    mdl = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    mdl.eval()
    return tok, mdl

def generate_poem(tok, mdl, prompt, max_len=120):
    """시 생성"""
    ids = tok.encode(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        ids, max_length=max_len, do_sample=True,
        top_k=50, top_p=0.95, repetition_penalty=1.2,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
    )
    return tok.decode(out[0].tolist(), skip_special_tokens=True)

def save_poem_to_file(poem: str, image_path: str, vision_results: List[Tuple[str, float]], 
                     output_dir: Path) -> str:
    """시를 파일로 저장"""
    output_dir.mkdir(exist_ok=True)
    
    # 파일명 생성 (타임스탬프 + 이미지명)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    filename = f"poem_{timestamp}_{image_name}.txt"
    output_path = output_dir / filename
    
    # 메타데이터와 함께 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("Generated Poem\n")
        f.write("=" * 50 + "\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Vision Results: {', '.join([f'{ko}({score:.3f})' for ko, score in vision_results])}\n")
        f.write("=" * 50 + "\n\n")
        f.write(poem)
        f.write("\n\n" + "=" * 50 + "\n")
    
    print(f"[INFO] Poem saved to: {output_path}")
    return str(output_path)

def vision_to_poem_pipeline(image_path: str, labels_file: str, kogpt2_path: str,
                          topk: int = 3, max_len: int = 120) -> str:
    """Vision → Poem 파이프라인"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Vision 분석
    print("\n[Vision Analysis]")
    clf = ClipKoreanClassifier(labels_file, DEFAULT_CLIP_MODEL, device=device.type)
    vis_results = clf.classify(image_path, topk=topk)
    
    for ko, sc in vis_results:
        print(f"  • {ko}  [{sc:.3f}]")
    
    # 프롬프트 생성
    prompt = ", ".join([ko for ko, _ in vis_results])
    print(f"\n[Prompt] {prompt}")
    
    # 시 생성
    print("\n[Poem Generation]")
    tok, mdl = load_kogpt2(kogpt2_path, device)
    poem = generate_poem(tok, mdl, prompt, max_len)
    
    print("\n[Generated Poem]")
    print("-" * 30)
    print(poem)
    print("-" * 30)
    
    # 파일 저장
    output_path = save_poem_to_file(poem, image_path, vis_results, OUTPUT_DIR)
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision to Poem Generator")
    parser.add_argument("--image_path", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--kogpt2_path", default=str(DEFAULT_KOGPT2_PATH))
    parser.add_argument("--labels_file", default=str(DEFAULT_LABELS_FILE))
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=120)
    args = parser.parse_args()
    
    # 파이프라인 실행
    output_file = vision_to_poem_pipeline(
        image_path=args.image_path,
        labels_file=args.labels_file,
        kogpt2_path=args.kogpt2_path,
        topk=args.topk,
        max_len=args.max_len
    )
    
    print(f"\n[Complete] Output saved to: {output_file}")
