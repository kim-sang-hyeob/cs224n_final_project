#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViT-GPT2 기반 이미지 캡셔닝/분류기
-------------------------------------------------------------
1) HuggingFace VisionEncoderDecoderModel 로드
2) ViTImageProcessor, GPT2 Tokenizer 로드
3) generate_caption(img) ─ 이미지 하나에 대해 캡션 생성
4) main()              ─ CLI: 파일 또는 폴더 처리
"""

import argparse, sys, torch
from pathlib import Path
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)

def load_model(model_name: str, device: torch.device):
    print(f"[INFO] Loading model {model_name} on {device}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, processor, tokenizer

@torch.no_grad()
def generate_caption(
    image: Image.Image,
    model: VisionEncoderDecoderModel,
    processor: ViTImageProcessor,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 16,
    num_beams: int = 4
) -> str:
    # 1) 이미지 전처리: pixel_values tensor
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    # 2) generate() 호출
    output_ids = model.generate(
        pixel_values,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    # 3) 토큰 디코딩
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

def get_image_files(path: Path):
    if path.is_file():
        return [path]
    elif path.is_dir():
        exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.tiff','*.webp']
        files = []
        for ext in exts:
            files += list(path.rglob(ext)) + list(path.rglob(ext.upper()))
        return sorted(files)
    else:
        return []

def main():
    parser = argparse.ArgumentParser(
        description="ViT-GPT2 Image Captioning",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("path", help="이미지 파일 또는 디렉터리 경로")
    parser.add_argument(
        "--model", default="nlpconnect/vit-gpt2-image-captioning",
        help="VisionEncoderDecoder 모델 이름 (기본: nlpconnect/vit-gpt2-image-captioning)"
    )
    parser.add_argument(
        "--max-length", type=int, default=16,
        help="생성 캡션 최대 길이 (기본: 16 토큰)"
    )
    parser.add_argument(
        "--beams", type=int, default=4,
        help="Beam search 빔 수 (기본: 4)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, tokenizer = load_model(args.model, device)

    files = get_image_files(Path(args.path))
    if not files:
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {args.path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {len(files)}개 이미지 처리 시작...\n")
    for img_path in files:
        try:
            img = Image.open(img_path).convert("RGB")
            caption = generate_caption(
                img, model, processor, tokenizer, device,
                max_length=args.max_length, num_beams=args.beams
            )
            print(f"{img_path.name:<30} → {caption}")
        except Exception as e:
            print(f"[WARNING] 처리 실패 ({img_path.name}): {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
