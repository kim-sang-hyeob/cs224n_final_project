#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision → Poetry : CLIP + KoGPT-2
usage:
    python generate_poem.py --image_path img.jpg --out_txt poem.txt
"""
import os, sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch, torch.nn.functional as F
from PIL import Image
from transformers import (CLIPModel, CLIPProcessor,
                          PreTrainedTokenizerFast, GPT2LMHeadModel)

# ─────────── 경로·상수 ───────────
PROJ_ROOT      = Path(__file__).resolve().parent
DEFAULT_CLIP   = "openai/clip-vit-base-patch32"
DEFAULT_GPT2   = PROJ_ROOT / "best_models" / "freeze-epoch5"
TOPICS_TXT     = PROJ_ROOT / "topics.txt"

ko2en = {...}            # ← 기존 딕셔너리 그대로 복붙

def load_labels(fp: Path) -> List[str]:
    return [ln.strip() for ln in fp.open(encoding="utf-8") if ln.strip()]

class ClipKoClassifier:
    def __init__(self, lbl_file: Path, model_name=DEFAULT_CLIP, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = CLIPModel.from_pretrained(model_name).to(self.device)
        self.proc   = CLIPProcessor.from_pretrained(model_name)

        ko = load_labels(lbl_file)
        self.ko, self.en = [], []
        for k in ko:
            if (e := ko2en.get(k)):
                self.ko.append(k); self.en.append(e)

        txt = [f"a photo of {e}" for e in self.en]
        enc = self.proc(text=txt, return_tensors="pt",
                        padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            self.t_emb = F.normalize(self.model.get_text_features(**enc), dim=-1)

    @torch.no_grad()
    def classify(self, img_path: Path, topk=3) -> List[Tuple[str,float]]:
        img = Image.open(img_path).convert("RGB")
        enc = self.proc(images=img, return_tensors="pt").to(self.device)
        i_emb = F.normalize(self.model.get_image_features(**enc), dim=-1)
        sims  = (i_emb @ self.t_emb.T)[0]
        sc, ix = sims.topk(min(topk, len(self.ko)))
        return [(self.ko[i.item()], sc[j].item()) for j,i in enumerate(ix)]

def load_kogpt2(model_dir: Path, device):
    tok = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
        use_fast=False)
    mdl = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    mdl.eval()
    return tok, mdl

def generate(tok, mdl, prompt, max_len=120):
    ids = tok.encode(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(ids, max_length=max_len, do_sample=True,
                       top_k=50, top_p=0.95, repetition_penalty=1.2,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

# ─────────── main ───────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--out_txt",   default="poem.txt")
    ap.add_argument("--gpt2_path", default=str(DEFAULT_GPT2))
    ap.add_argument("--labels",    default=str(TOPICS_TXT))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=120)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = ClipKoClassifier(Path(args.labels), device=dev.type)
    topics = clf.classify(Path(args.image_path), topk=args.topk)
    prompt = ", ".join([k for k,_ in topics])

    tok, mdl = load_kogpt2(Path(args.gpt2_path), dev)
    poem = generate(tok, mdl, prompt, max_len=args.max_len).strip()

    Path(args.out_txt).write_text(poem, encoding="utf-8")
    print(f"[saved] {args.out_txt}")
