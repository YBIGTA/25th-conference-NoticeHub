
from fastapi import FastAPI, HTTPException
import subprocess
import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from pydantic import BaseModel
from dotenv import load_dotenv
import json
# FastAPI 애플리케이션 생성
app = FastAPI()
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# .env 파일 경로 명시 (기본은 현재 디렉토리)
load_dotenv(dotenv_path="../.env")
# 환경 변수 설정 (S3 인증)
os.environ["AWS_ACCESS_KEY_ID"] =os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] =os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_BUCKET_NAME"] =os.getenv("AWS_BUCKET_NAME")

# Pydantic 모델 (HTTP 요청 데이터 검증용)
class S3Urls(BaseModel):
    urls: list[str]


@app.post("/process_image/")
async def process_image(request: S3Urls):
    """
    HTTP POST 요청으로 S3 URL 리스트를 받아 subprocess로 처리 스크립트를 실행.
    """
    try:
        # 전달받은 S3 URL 리스트
        s3_urls = request.urls

        # JSON 문자열로 변환
        s3_urls_json = json.dumps(s3_urls)

        # JSON 배열을 subprocess로 전달
        logger.info(f"Received S3 URLs: {s3_urls}")
        subprocess.run(
            ["python", os.path.abspath("../testing_multi.py"), s3_urls_json],  # 리스트를 문자열로 변환하여 전달
            check=True
        )
        return {"status": "success", "message": "Image processing initiated"}
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error while running testing_multi.py: {e}")
        raise HTTPException(status_code=500, detail="Failed to process images")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")