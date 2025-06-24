#!/usr/bin/env python3
"""
CLIP 토큰 추출 함수 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision.CLIP import extract_clip_tokens

def test_clip_extraction():
    """CLIP 토큰 추출 함수 테스트"""
    
    # 테스트할 이미지 경로 (실제 이미지 파일이 필요합니다)
    # 예시: "test_image.jpg" 또는 실제 이미지 파일 경로
    test_image_path = "test_image.jpg"  # 실제 이미지 파일로 변경 필요
    
    try:
        print("=== CLIP 토큰 추출 테스트 ===")
        print(f"이미지 경로: {test_image_path}")
        
        # CLIP 토큰 추출
        extracted_tokens = extract_clip_tokens(test_image_path)
        
        print(f"\n추출된 토큰: {extracted_tokens}")
        print("\n=== 테스트 완료 ===")
        
        return extracted_tokens
        
    except FileNotFoundError:
        print(f"이미지 파일을 찾을 수 없습니다: {test_image_path}")
        print("실제 이미지 파일 경로로 변경해주세요.")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    test_clip_extraction() 