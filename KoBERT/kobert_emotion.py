#!/usr/bin/env python3
"""
KoBERT 감정 분류 모듈
사용자입력.py를 기반으로 import 가능한 형태로 변환
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class KoBERTEmotionClassifier:
    def __init__(self, model_dir: str = None, device: str = None):
        """
        KoBERT 감정 분류기 초기화
        
        Args:
            model_dir (str): 모델 디렉터리 경로
            device (str): 사용할 디바이스 (cuda/cpu)
        """
        # 기본 모델 경로 설정
        if model_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, "kobert-emotion")
        
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 감정 레이블 매핑
        self.id2label = {
            0: "분노",
            1: "슬픔", 
            2: "불안",
            3: "당황",
            4: "상처",
            5: "무감정",
            6: "기쁨",
        }
        
        print(f"[INFO] Loading KoBERT emotion model from {model_dir} on {self.device}...")
        
        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] KoBERT emotion model loaded successfully on {self.device}")
    
    def predict_emotion(self, text: str):
        """
        입력 문장에 대해 감정을 분류합니다.
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            tuple: (예측된 감정, 모든 감정의 확률 리스트)
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("입력 문장은 비어 있을 수 없습니다.")

        # 토큰화
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # 어휘 크기 체크
        vocab_size = self.tokenizer.vocab_size
        max_id = int(input_ids.max().item())
        if max_id >= vocab_size:
            raise IndexError(
                f"토큰화된 입력에 어휘 크기(vocab_size={vocab_size})를 초과하는 인덱스가 있습니다: max_id={max_id}"
            )

        # 모델 추론
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # 확률이 높은 순서로 정렬
        sorted_indices = np.argsort(probs)[::-1]
        scores = [(self.id2label[i], float(probs[i])) for i in sorted_indices]

        pred_label = self.id2label[sorted_indices[0]]
        return pred_label, scores
    
    def safe_predict_emotion(self, text: str):
        """
        예외 처리가 포함된 안전한 감정 예측 함수
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            tuple: (예측된 감정, 확률 리스트) 또는 (None, None) if 오류
        """
        try:
            return self.predict_emotion(text)
        except Exception as e:
            print(f"[오류] 감정 예측 중 예외 발생: {e}")
            return None, None
    
    def analyze_poem_emotion(self, poem: str, show_details: bool = True):
        """
        시의 감정을 분석하는 특화된 함수
        
        Args:
            poem (str): 분석할 시
            show_details (bool): 상세 정보 출력 여부
            
        Returns:
            dict: 감정 분석 결과
        """
        if show_details:
            print(f"=== 시 감정 분석 ===")
            print(f"시: {poem[:100]}{'...' if len(poem) > 100 else ''}")
        
        emotion, scores = self.safe_predict_emotion(poem)
        
        if emotion is None:
            return {
                "emotion": "분석 실패",
                "confidence": 0.0,
                "all_emotions": [],
                "poem": poem
            }
        
        result = {
            "emotion": emotion,
            "confidence": scores[0][1] if scores else 0.0,
            "all_emotions": scores,
            "poem": poem
        }
        
        if show_details:
            print(f"주요 감정: {emotion} (신뢰도: {result['confidence']:.3f})")
            print("전체 감정 분포:")
            for emotion_name, prob in scores:
                print(f"  - {emotion_name}: {prob:.3f}")
            print()
        
        return result

# 편의 함수들
def create_emotion_classifier(model_dir: str = None, device: str = None) -> KoBERTEmotionClassifier:
    """
    KoBERT 감정 분류기 인스턴스를 생성하는 편의 함수
    """
    return KoBERTEmotionClassifier(model_dir, device)

def classify_emotion(text: str) -> str:
    """
    간단한 감정 분류 함수 (기본 설정 사용)
    
    Args:
        text (str): 분석할 텍스트
        
    Returns:
        str: 예측된 감정
    """
    classifier = create_emotion_classifier()
    emotion, _ = classifier.safe_predict_emotion(text)
    return emotion or "분석 실패"

# 테스트 함수
def test_emotion_classification():
    """감정 분류 기능 테스트"""
    print("=== KoBERT 감정 분류 테스트 ===")
    
    # 테스트용 시들
    test_poems = [
        "오늘은 정말 기쁜 날이에요. 햇살이 따뜻하고 꽃들이 예쁘게 피어있어요.",
        "비가 내리는 창가에서 혼자 앉아있자니 마음이 슬퍼져요.",
        "갑자기 무서운 일이 일어날 것 같아서 불안해요.",
        "예상치 못한 일이 벌어져서 당황스러워요."
    ]
    
    try:
        # 감정 분류기 생성
        classifier = create_emotion_classifier()
        
        # 각 시의 감정 분석
        for i, poem in enumerate(test_poems, 1):
            print(f"\n--- 테스트 시 {i} ---")
            result = classifier.analyze_poem_emotion(poem, show_details=True)
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_emotion_classification() 