#!/usr/bin/env python3
"""
KoGPT2 시 생성 모듈
NLP_No_tuning.ipynb의 핵심 기능을 Python 스크립트로 변환
"""

import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

class KoGPT2PoetryGenerator:
    def __init__(self, model_name: str = "skt/kogpt2-base-v2", device: str = None):
        """
        KoGPT2 시 생성기 초기화
        
        Args:
            model_name (str): 사용할 KoGPT2 모델 이름
            device (str): 사용할 디바이스 (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading KoGPT2 model ({model_name}) on {self.device}...")
        
        # 토크나이저 로드
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_name,
            bos_token='</s>', eos_token='</s>',
            unk_token='<unk>', pad_token='<pad>', mask_token='<mask>'
        )
        
        # 모델 로드
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 생성 모드로 변경 (dropout off)
        
        print(f"[INFO] KoGPT2 model loaded successfully on {self.device}")
    
    def generate_korean_text(self, 
                           prompt: str,
                           max_length: int = 500,
                           top_k: int = 50,
                           top_p: float = 0.95,
                           repetition_penalty: float = 1.2,
                           temperature: float = 1.0) -> str:
        """
        한국어 텍스트 생성 함수
        
        Args:
            prompt (str): 생성 시작 문자열 (예: "아름다운 풍경에 대한 시:")
            max_length (int): 생성할 최대 토큰 길이
            top_k (int): top-k sampling 개수
            top_p (float): nucleus sampling 확률 임계치
            repetition_penalty (float): 반복 패널티
            temperature (float): 생성 온도 (높을수록 다양성 증가)
        
        Returns:
            str: 생성된 문자열 (prompt + continuation)
        """
        # 입력 문장 토큰화
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # generate() 호출
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=1,             # 빔 서치 사용 안 함
                do_sample=True,          # 샘플링 모드
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3   # 3-gram 반복 방지
            )
        
        # 토큰 시퀀스를 문자열로 디코딩
        generated = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        return generated
    
    def generate_poetry_from_tokens(self, 
                                  tokens: str,
                                  max_length: int = 100,
                                  style: str = "modern") -> str:
        """
        CLIP에서 추출한 토큰을 바탕으로 시를 생성하는 함수
        
        Args:
            tokens (str): CLIP에서 추출한 토큰 (예: "아름다운 풍경, 자연과 평온")
            max_length (int): 생성할 최대 길이
            style (str): 시 스타일 ("modern", "traditional", "romantic")
        
        Returns:
            str: 생성된 시
        """
        # 스타일에 따른 프롬프트 생성
        style_prompts = {
            "modern": f"다음 키워드로 현대시를 써주세요: {tokens}\n",
            "traditional": f"다음 키워드로 전통적인 시를 써주세요: {tokens}\n",
            "romantic": f"다음 키워드로 로맨틱한 시를 써주세요: {tokens}\n"
        }
        
        prompt = style_prompts.get(style, style_prompts["modern"])
        
        # 시 생성
        generated_poem = self.generate_korean_text(
            prompt=prompt,
            max_length=max_length,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            temperature=0.8
        )
        
        return generated_poem
    
    def generate_multiple_poems(self, 
                              tokens: str, 
                              num_poems: int = 3,
                              max_length: int = 100) -> list:
        """
        여러 개의 시를 생성하는 함수
        
        Args:
            tokens (str): CLIP에서 추출한 토큰
            num_poems (int): 생성할 시의 개수
            max_length (int): 각 시의 최대 길이
        
        Returns:
            list: 생성된 시들의 리스트
        """
        poems = []
        styles = ["modern", "traditional", "romantic"]
        
        for i in range(num_poems):
            style = styles[i % len(styles)]
            poem = self.generate_poetry_from_tokens(tokens, max_length, style)
            poems.append({
                "style": style,
                "poem": poem,
                "index": i + 1
            })
        
        return poems

# 편의 함수들
def create_poetry_generator(model_name: str = "skt/kogpt2-base-v2", device: str = None) -> KoGPT2PoetryGenerator:
    """
    KoGPT2 시 생성기 인스턴스를 생성하는 편의 함수
    """
    return KoGPT2PoetryGenerator(model_name, device)

def generate_poetry_simple(tokens: str, max_length: int = 100) -> str:
    """
    간단한 시 생성 함수 (기본 설정 사용)
    """
    generator = create_poetry_generator()
    return generator.generate_poetry_from_tokens(tokens, max_length)

# 테스트 함수
def test_poetry_generation():
    """시 생성 기능 테스트"""
    print("=== KoGPT2 시 생성 테스트 ===")
    
    # 테스트용 토큰 (CLIP에서 추출된 것처럼)
    test_tokens = "아름다운 풍경, 자연과 평온, 빛과 그림자"
    
    try:
        # 시 생성기 생성
        generator = create_poetry_generator()
        
        # 단일 시 생성
        print(f"\n입력 토큰: {test_tokens}")
        print("\n--- 생성된 시 ---")
        poem = generator.generate_poetry_from_tokens(test_tokens, max_length=80)
        print(poem)
        
        # 여러 스타일의 시 생성
        print("\n--- 다양한 스타일의 시 ---")
        poems = generator.generate_multiple_poems(test_tokens, num_poems=2, max_length=80)
        for poem_info in poems:
            print(f"\n[{poem_info['style']} 스타일]")
            print(poem_info['poem'])
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_poetry_generation() 