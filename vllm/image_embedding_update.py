from byaldi import RAGMultiModalModel
from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Union
import shutil  # 이 줄 추가
import torch
from PIL import Image
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from byaldi.colpali import ColPaliModel
from byaldi.objects import Result
import logging
import sys
import json
from io import BytesIO
from pymongo import MongoClient
# Optional langchain integration
try:
    from byaldi.integrations import ByaldiLangChainRetriever
except ImportError:
    pass

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# MongoDB 설정
MONGO_PATH = os.getenv("MONGO_PATH")
client = MongoClient(MONGO_PATH)
db = client["notice-db"]
collection = db["final_notices"]

# S3 클라이언트 설정
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

def parse_s3_object_url(s3_object_url: str):
    """
    S3 객체의 HTTPS URL에서 버킷 이름과 객체 키를 추출합니다.
    
    Args:
        s3_object_url (str): S3 객체의 HTTPS URL (예: https://bucket-name.s3.amazonaws.com/path/to/image.jpg)
    
    Returns:
        tuple: (bucket_name, object_key)
    """
    if not s3_object_url.startswith("https://"):
        raise ValueError(f"Invalid S3 URL: {s3_object_url}")
    
    # 'https://' 이후의 부분을 추출
    url_body = s3_object_url[8:]
    
    # 버킷 이름과 나머지 부분을 분리
    bucket_and_rest = url_body.split(".s3.amazonaws.com/", 1)
    if len(bucket_and_rest) != 2:
        raise ValueError(f"Invalid S3 URL format: {s3_object_url}")
    
    bucket_name = bucket_and_rest[0]
    object_key = bucket_and_rest[1]
    
    return bucket_name, object_key

def download_from_s3(s3_object_url: str):
    """
    S3 URL로부터 이미지를 다운로드하여 메모리 상에서 처리.
    
    Args:
        s3_object_url: S3 URL (예: https://bucket-name.s3.amazonaws.com/path/to/image.jpg)
    """
    try:
        bucket_name, object_key = parse_s3_object_url(s3_object_url)
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Failed to download image from S3: {e}")
        return None

class RAGMultiModalModel:
    """
    Wrapper class for a pretrained RAG multi-modal model, and all the associated utilities.
    Allows you to load a pretrained model from disk or from the hub, build or query an index.

    ## Usage

    Load a pre-trained checkpoint:

    ```python
    from byaldi import RAGMultiModalModel

    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
    ```

    Both methods will load a fully initialized instance of ColPali, which you can use to build and query indexes.

    ```python
    RAG.search("How many people live in France?")
    ```
    """

    model: Optional[ColPaliModel] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        index_root: str = ".byaldi",
        device: str = None,
        verbose: int = 1,
    ):
        """Load a ColPali model from a pre-trained checkpoint.

        Parameters:
            pretrained_model_name_or_path (str): Local path or Hugging Face model name.
            device (str): The device to load the model on. Default is "cuda".

        Returns:
            cls (RAGMultiModalModel): The current instance of RAGMultiModalModel, with the model initialized.
        """
        instance = cls()
         # 디바이스 설정 (기본값: "cpu", GPU가 있으면 "cuda")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        instance.model = ColPaliModel.from_pretrained(
            pretrained_model_name_or_path,
            index_root=index_root,
            device=device,
            verbose=verbose,
        )
        logger.info(f"모델이 {device}에서 로드되었습니다.")
        return instance

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        index_root: str = ".byaldi",
        device: str = None,
        verbose: int = 1,
    ):
        """Load an Index and the associated ColPali model from an existing document index.

        Parameters:
            index_path (Union[str, Path]): Path to the index.
            device (str): The device to load the model on. Default is "cuda".

        Returns:
            cls (RAGMultiModalModel): The current instance of RAGMultiModalModel, with the model and index initialized.
        """
        instance = cls()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        index_path = Path(index_path)
        instance.model = ColPaliModel.from_index(
            index_path, index_root=index_root, device=device, verbose=verbose
        )
        logger.info(f"모델이 {device}에서 로드되었습니다.")
        return instance

    def index(
        self,
        input_path: Union[str, Path],
        index_name: Optional[str] = None,
        doc_ids: Optional[List[Union[str, int]]] = None,
        store_collection_with_index: bool = False,
        overwrite: bool = False,
        metadata: Optional[
            Union[
                Dict[Union[str, int], Dict[str, Union[str, int]]],
                List[Dict[str, Union[str, int]]],
            ]
        ] = None,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
        **kwargs,
    ):
        """Build an index from input documents.

        Parameters:
            input_path (Union[str, Path]): Path to the input documents.
            index_name (Optional[str]): The name of the index that will be built.
            doc_ids (Optional[List[Union[str, int]]]): List of document IDs.
            store_collection_with_index (bool): Whether to store the collection with the index.
            overwrite (bool): Whether to overwrite an existing index with the same name.
            metadata (Optional[Union[Dict[Union[str, int], Dict[str, Union[str, int]]], List[Dict[str, Union[str, int]]]]]):
                Metadata for the documents. Can be a dictionary mapping doc_ids to metadata dictionaries,
                or a list of metadata dictionaries (one for each document).

        Returns:
            None
        """
        return self.model.index(
            input_path,
            index_name,
            doc_ids,
            store_collection_with_index,
            overwrite=overwrite,
            metadata=metadata,
            max_image_width=max_image_width,
            max_image_height=max_image_height,
            **kwargs,
        )

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image],
        store_collection_with_index: bool,
        doc_id: Optional[Union[str, int]] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        """Add an item to an existing index.

        Parameters:
            input_item (Union[str, Path, Image.Image]): The item to add to the index.
            store_collection_with_index (bool): Whether to store the collection with the index.
            doc_id (Union[str, int]): The document ID for the item being added.
            metadata (Optional[Dict[str, Union[str, int]]]): Metadata for the document being added.

        Returns:
            None
        """
        return self.model.add_to_index(
            input_item, store_collection_with_index, doc_id, metadata=metadata
        )

class MultiModalRAG(RAGMultiModalModel):
    _instance = None  # 싱글톤 패턴을 위한 클래스 변수
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 환경 변수 로드
            load_dotenv(dotenv_path="/app/.env")
            
            # 환경 변수 설정
            os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            
            # RAG 모델 초기화 (한 번만 실행됨)
            logger.info("RAG 모델 초기화 중...")
            cls._instance.model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1).model
            
        return cls._instance

    def create_index(self, images: List[Image.Image], is_initial: bool = True):
        """이미지 리스트로부터 인덱스 생성 및 임베딩 정보 반환
        
        Args:
            images: 인덱싱할 이미지 리스트
            metadata: 이미지에 대한 메타데이터 리스트
            is_initial: 초기 인덱스 생성 여부
        
        Returns:
            dict: 각 이미지의 임베딩, 메타데이터, doc_id를 포함하는 정보
        """
        try:
            logger.info(f"인덱싱 시작 (초기 인덱스 생성: {is_initial})")
            
            if is_initial:
                temp_dir = "temp_images"
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    # 이미지들을 임시 디렉토리에 저장
                    for idx, image in enumerate(images):
                        image_path = os.path.join(temp_dir, f"image_{idx}.png")
                        image.save(image_path)
                    
                    # 인덱스 생성
                    self.model.index(
                        input_path=temp_dir,
                        index_name="image_index",
                        store_collection_with_index=True,
                        overwrite=is_initial
                    )
                finally:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            else:
                # 기존 인덱스에 추가
                for idx, image in enumerate(images):
                    self.model.add_to_index(
                        input_item=image,
                        store_collection_with_index=True,
                        doc_id=str(idx)
                    )
            
            # 이미지 임베딩 생성
            embeddings = self.model.encode_image(images)
            
            # MongoDB에 저장할 형태로 결과 구성
            result = []
            for idx, (embedding, image) in enumerate(zip(embeddings, images)):
                
                # bfloat16을 float32로 변환 후 리스트로 변환
                embedding_list = embedding.to(torch.float32).detach().cpu().numpy().tolist()
                
                result.append({
                    "doc_id": str(idx),
                    "image_embedding": embedding_list,
                    "status": "indexed"
                })
            
            logger.info(f"인덱싱 완료: {len(images)}개 이미지 처리됨")
            return result
            
        except Exception as e:
            logger.error(f"인덱싱 중 오류 발생: {str(e)}")
            raise

def update_mongodb(collection, index_info):
    """
    MongoDB 데이터를 업데이트하는 함수.
    특정 `images` 필드를 기준으로 데이터를 검색하여 embedding과 기타 데이터를 추가.

    Args:
        collection: MongoDB 컬렉션
        index_info: 생성된 인덱스 정보 (doc_id, image_embedding, metadata, status 포함)
    """
    try:
        for record in index_info:
            # 이미지 파일 이름을 기준으로 기존 document 검색
            image_filename = record["metadata"].get("filename")
            if not image_filename:
                logger.warning("메타데이터에 filename이 없습니다. 건너뜁니다.")
                continue
            
            # MongoDB에서 해당하는 document 찾기
            doc = collection.find_one({"images": f"https://noticehub.s3.amazonaws.com/images/{image_filename}"})
            if not doc:
                logger.warning(f"'{image_filename}'을 포함하는 document를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # MongoDB document 업데이트
            collection.update_one(
                {"_id": doc["_id"]},  # 해당 document의 `_id` 기준
                {"$set": {"image_embedding": record["image_embedding"]}}
            )
            
            logger.info(f"Document 업데이트 완료: {doc['_id']} (이미지 파일: {image_filename})")
    
    except Exception as e:
        logger.error(f"MongoDB 업데이트 중 오류 발생: {str(e)}")
        raise


def main():
    try:
        # RAG 모델 초기화
        rag = MultiModalRAG()
        logger.info("RAG 모델 호출 중...")
        
        documents = collection.find({"images": {"$exists": True}})
        for doc in documents:
            s3_object_url = doc["images"]
            logger.info(f"Processing document {doc['_id']} with image {s3_object_url}")
            image = download_from_s3(s3_object_url)
            if image:
                # 인덱스 생성 및 결과 반환
                index_info = rag.create_index(image, is_initial=True)  # 테스트를 위해 초기 인덱스 생성
                logger.info("인덱싱 완료")

                # MongoDB 업데이트
                logger.info("MongoDB 업데이트 시작...")
                update_mongodb(collection, index_info)
                logger.info("MongoDB 업데이트 완료")
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()