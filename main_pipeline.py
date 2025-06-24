#!/usr/bin/env python3
"""
이미지 → CLIP 토큰 추출 → KoGPT2 시 생성 → KoBERT 감정 분류
전체 파이프라인
"""

import os
import sys
import time
import torch
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from vision.CLIP import extract_clip_tokens
from kogpt2.kogpt2_generate import create_poetry_generator
from KoBERT.kobert_emotion import create_emotion_classifier

class ImageToPoetryPipeline:
    def __init__(self, device: str = None):
        """
        이미지 → 시 생성 파이프라인 초기화
        
        Args:
            device (str): 사용할 디바이스 (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing pipeline on {self.device}...")
        
        # 각 모듈 초기화
        self.clip_ready = False
        self.kogpt2_ready = False
        self.kobert_ready = False
        
        print("[INFO] Pipeline initialized successfully!")
    
    def load_clip(self):
        """CLIP 모듈 로드 (지연 로딩)"""
        if not self.clip_ready:
            print("[INFO] Loading CLIP module...")
            # CLIP은 함수 호출 시 로드되므로 여기서는 준비 상태만 표시
            self.clip_ready = True
            print("[INFO] CLIP module ready!")
    
    def load_kogpt2(self):
        """KoGPT2 모듈 로드 (지연 로딩)"""
        if not self.kogpt2_ready:
            print("[INFO] Loading KoGPT2 module...")
            self.poetry_generator = create_poetry_generator(device=self.device)
            self.kogpt2_ready = True
            print("[INFO] KoGPT2 module loaded!")
    
    def load_kobert(self):
        """KoBERT 모듈 로드 (지연 로딩)"""
        if not self.kobert_ready:
            print("[INFO] Loading KoBERT module...")
            self.emotion_classifier = create_emotion_classifier(device=self.device)
            self.kobert_ready = True
            print("[INFO] KoBERT module loaded!")
    
    def process_image(self, 
                     image_path: str, 
                     poetry_style: str = "modern",
                     max_length: int = 100,
                     num_poems: int = 1,
                     show_details: bool = True) -> dict:
        """
        이미지를 처리하여 시를 생성하고 감정을 분석하는 메인 함수
        
        Args:
            image_path (str): 입력 이미지 경로
            poetry_style (str): 시 스타일 ("modern", "traditional", "romantic")
            max_length (int): 생성할 시의 최대 길이
            num_poems (int): 생성할 시의 개수
            show_details (bool): 상세 정보 출력 여부
            
        Returns:
            dict: 전체 처리 결과
        """
        start_time = time.time()
        
        if show_details:
            print("=" * 60)
            print("🎨 이미지 → 시 생성 파이프라인 시작")
            print("=" * 60)
            print(f"📸 입력 이미지: {image_path}")
            print(f"🎭 시 스타일: {poetry_style}")
            print(f"📏 최대 길이: {max_length}")
            print(f"📝 생성할 시 개수: {num_poems}")
            print()
        
        try:
            # 1단계: CLIP 토큰 추출
            if show_details:
                print("🔍 1단계: CLIP으로 이미지 분석 중...")
            
            self.load_clip()
            clip_start = time.time()
            extracted_tokens = extract_clip_tokens(image_path, device=self.device)
            clip_time = time.time() - clip_start
            
            if show_details:
                print(f"✅ 추출된 토큰: {extracted_tokens}")
                print(f"⏱️  CLIP 처리 시간: {clip_time:.2f}초")
                print()
            
            # 2단계: KoGPT2 시 생성
            if show_details:
                print("✍️ 2단계: KoGPT2로 시 생성 중...")
            
            self.load_kogpt2()
            kogpt2_start = time.time()
            
            if num_poems == 1:
                # 단일 시 생성
                poem = self.poetry_generator.generate_poetry_from_tokens(
                    extracted_tokens, max_length, poetry_style
                )
                poems = [{"style": poetry_style, "poem": poem, "index": 1}]
            else:
                # 여러 시 생성
                poems = self.poetry_generator.generate_multiple_poems(
                    extracted_tokens, num_poems, max_length
                )
            
            kogpt2_time = time.time() - kogpt2_start
            
            if show_details:
                print(f"✅ {len(poems)}개의 시 생성 완료")
                print(f"⏱️  KoGPT2 처리 시간: {kogpt2_time:.2f}초")
                print()
            
            # 3단계: KoBERT 감정 분류
            if show_details:
                print("🧠 3단계: KoBERT로 감정 분석 중...")
            
            self.load_kobert()
            kobert_start = time.time()
            
            emotion_results = []
            for poem_info in poems:
                emotion_result = self.emotion_classifier.analyze_poem_emotion(
                    poem_info["poem"], show_details=False
                )
                emotion_results.append(emotion_result)
            
            kobert_time = time.time() - kobert_start
            
            if show_details:
                print(f"✅ 감정 분석 완료")
                print(f"⏱️  KoBERT 처리 시간: {kobert_time:.2f}초")
                print()
            
            # 전체 결과 정리
            total_time = time.time() - start_time
            
            result = {
                "input_image": image_path,
                "extracted_tokens": extracted_tokens,
                "poems": poems,
                "emotion_analysis": emotion_results,
                "processing_times": {
                    "clip": clip_time,
                    "kogpt2": kogpt2_time,
                    "kobert": kobert_time,
                    "total": total_time
                },
                "settings": {
                    "style": poetry_style,
                    "max_length": max_length,
                    "num_poems": num_poems
                }
            }
            
            if show_details:
                print("=" * 60)
                print("🎉 파이프라인 완료!")
                print("=" * 60)
                print(f"⏱️  총 처리 시간: {total_time:.2f}초")
                print()
                
                # 결과 요약 출력
                self._print_summary(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 파이프라인 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _print_summary(self, result: dict):
        """결과 요약 출력"""
        print("📊 결과 요약:")
        print(f"  • 추출된 토큰: {result['extracted_tokens']}")
        print(f"  • 생성된 시 개수: {len(result['poems'])}")
        
        for i, (poem_info, emotion_result) in enumerate(zip(result['poems'], result['emotion_analysis'])):
            print(f"\n  📝 시 {i+1} ({poem_info['style']} 스타일):")
            print(f"    감정: {emotion_result['emotion']} (신뢰도: {emotion_result['confidence']:.3f})")
            poem_preview = poem_info['poem'][:50] + "..." if len(poem_info['poem']) > 50 else poem_info['poem']
            print(f"    내용: {poem_preview}")
        
        print(f"\n⏱️  처리 시간:")
        print(f"  • CLIP: {result['processing_times']['clip']:.2f}초")
        print(f"  • KoGPT2: {result['processing_times']['kogpt2']:.2f}초")
        print(f"  • KoBERT: {result['processing_times']['kobert']:.2f}초")
        print(f"  • 총합: {result['processing_times']['total']:.2f}초")

# 편의 함수들
def create_pipeline(device: str = None) -> ImageToPoetryPipeline:
    """파이프라인 인스턴스 생성"""
    return ImageToPoetryPipeline(device)

def process_image_simple(image_path: str, style: str = "modern") -> dict:
    """간단한 이미지 처리 함수"""
    pipeline = create_pipeline()
    return pipeline.process_image(image_path, style, show_details=True)

# 테스트 함수
def test_pipeline():
    """전체 파이프라인 테스트"""
    print("=== 이미지 → 시 생성 파이프라인 테스트 ===")
    
    # 테스트할 이미지 경로 (실제 이미지 파일로 변경 필요)
    test_image_path = "test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"⚠️  테스트 이미지가 없습니다: {test_image_path}")
        print("실제 이미지 파일 경로로 변경해주세요.")
        return
    
    try:
        # 파이프라인 생성 및 실행
        pipeline = create_pipeline()
        result = pipeline.process_image(
            image_path=test_image_path,
            poetry_style="modern",
            max_length=80,
            num_poems=2,
            show_details=True
        )
        
        if result:
            print("\n✅ 파이프라인 테스트 성공!")
        else:
            print("\n❌ 파이프라인 테스트 실패!")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline() 