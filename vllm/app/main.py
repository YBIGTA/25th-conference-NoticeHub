
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess

# FastAPI 애플리케이션 생성
app = FastAPI()


# 데이터 모델 정의

class EmbeddingRequest(BaseModel):
    image_url: str


@app.post("/embedding")
async def embedding(request: EmbeddingRequest):

    """
    클라이언트 요청에서 이미지 URL을 받아 Docker로 모델 실행
    """
    try:
        # docker run 명령어 실행

        # 이미지 URL을 받아 Docker로 모델 실행
        result = subprocess.check_output(
            [
                "docker", "run", "--rm",
                "-e", f"IMAGE_URL={request.image_url}",
                "model-container-image"  # 모델이 패키징된 Docker 이미지 이름
            ],
            text=True  # 결과를 텍스트로 반환
        )

        return {"embedding": result.strip()}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))