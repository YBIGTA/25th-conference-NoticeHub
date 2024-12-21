from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# FastAPI 앱 초기화
app = FastAPI()

# 요청 데이터 모델
class PredictionRequest(BaseModel):
    prompt: str  # 클라이언트로부터 전달받을 프롬프트

# VLLM 서버 URL
VLLM_SERVER_URL = "http://vllm-server:8000/predict"

@app.post("/generate")
async def generate_text(request: PredictionRequest):
    """
    클라이언트 요청에서 프롬프트를 받아 VLLM 서버 호출 후 결과 반환
    """
    try:
        # VLLM 서버 호출
        vllm_response = requests.post(VLLM_SERVER_URL, json={"prompt": request.prompt})
        vllm_response.raise_for_status()  # HTTP 오류 발생 시 예외

        # 응답 성공 처리
        return {"status": "Model executed successfully"}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"VLLM Server request failed: {str(e)}")
