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

def get_all_s3_urls(bucket_name: str, prefix: str = "") -> List[str]:
    """
    S3 버킷의 특정 폴더(prefix)에 있는 모든 객체의 URL 리스트를 반환.
    
    Args:
        bucket_name (str): S3 버킷 이름.
        prefix (str): S3 객체 키의 접두사(폴더 경로).
    
    Returns:
        List[str]: S3 객체의 URL 리스트.
    """
    s3_urls = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    # S3 URL 생성
                    s3_url = f"s3://{bucket_name}/{key}"
                    s3_urls.append(s3_url)
        
        logger.info(f"{len(s3_urls)}개의 S3 URL을 찾았습니다.")
        return s3_urls
    
    except Exception as e:
        logger.error(f"S3 URL 가져오는 중 오류 발생: {e}")
        raise

def download_from_s3(s3_url: str, download_dir: str = "images") -> str:
    """
    S3 URL로부터 이미지를 다운로드하여 로컬 디렉토리에 저장.
    
    Args:
        s3_url: S3 URL
        download_dir: 이미지 저장 디렉토리
    
    Returns:
        str: 로컬 파일 경로
    """
    try:
        bucket_name, object_key = s3_url[5:].split("/", 1)
        os.makedirs(download_dir, exist_ok=True)
        local_file_path = os.path.join(download_dir, os.path.basename(object_key))
        s3_client.download_file(bucket_name, object_key, local_file_path)
        logger.info(f"Downloaded {object_key} to {local_file_path}")
        return local_file_path
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        raise

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

    def create_index(self, images: List[Image.Image], metadata: List[Dict] = None, is_initial: bool = True):
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
                        metadata=metadata,
                        overwrite=is_initial
                    )
                finally:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            else:
                # 기존 인덱스에 추가
                for idx, image in enumerate(images):
                    image_metadata = metadata[idx] if metadata else {"image_id": str(idx)}
                    self.model.add_to_index(
                        input_item=image,
                        store_collection_with_index=True,
                        doc_id=str(idx),
                        metadata=image_metadata
                    )
            
            # 이미지 임베딩 생성
            embeddings = self.model.encode_image(images)
            
            # MongoDB에 저장할 형태로 결과 구성
            result = []
            for idx, (embedding, image) in enumerate(zip(embeddings, images)):
                image_metadata = metadata[idx] if metadata else {"image_id": str(idx)}
                
                # bfloat16을 float32로 변환 후 리스트로 변환
                embedding_list = embedding.to(torch.float32).detach().cpu().numpy().tolist()
                
                result.append({
                    "doc_id": str(idx),
                    "image_embedding": embedding_list,
                    "metadata": image_metadata,
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
        index_info: 생성된 인덱스 정보 (doc_id, image_embedding, metadata 포함)
    """
    try:
        for record in index_info:
            image_filename = record["metadata"]["filename"]
            
            # "images" 필드에 {image_filename}이 존재하는지 확인
            doc = collection.find_one({"images": {"$regex": f".*{image_filename}$"}})
            if not doc:
                logger.warning(f"Document not found for {image_filename}. Skipping...")
                continue
            
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"image_embedding": record["image_embedding"]}}
            )
            logger.info(f"Document 업데이트 완료: {doc['_id']} (이미지 파일: {image_filename})")

    except Exception as e:
        logger.error(f"MongoDB update failed: {e}")
        raise

def main():
    try:
        # 1. S3에서 모든 URL 가져오기
        bucket_name = "noticehub"
        prefix = "images/"  # S3의 image 폴더
        logger.info(f"{bucket_name} 버킷에서 {prefix} 경로의 객체를 검색 중...")
        s3_urls = get_all_s3_urls(bucket_name, prefix)

        if not s3_urls:
            logger.error("S3에서 처리할 이미지 URL을 찾지 못했습니다.")
            return
        
        logger.info(f"총 {len(s3_urls)}개의 S3 URL이 수집되었습니다.")
        logger.debug(f"수집된 URL: {s3_urls}")

      # 2. 이미지 다운로드 및 로드
        image_dir = "images"
        images = []
        metadata = []
        for idx, s3_url in enumerate(s3_urls):
            try:
                local_path = download_from_s3(s3_url, download_dir=image_dir)
                image = Image.open(local_path)
                images.append(image)
                metadata.append({
                    "image_id": str(idx),
                    "filename": os.path.basename(local_path)
                })
            except Exception as e:
                logger.error(f"Error processing {s3_url}: {e}")
                continue

        if not images:
            logger.error("No images to process.")
            return

        # 3. RAG 모델 초기화
        rag = MultiModalRAG()
        logger.info("RAG 모델 호출 중...")

        # 4. 인덱스 생성 및 결과 반환
        logger.info(f"총 {len(images)}개 이미지 인덱싱 시작...")
        index_info = rag.create_index(images, metadata, is_initial=True)  # 테스트를 위해 초기 인덱스 생성
        logger.info("인덱싱 완료")
        # logger.info(f"생성된 인덱스 정보: {index_info}")

        # 5. MongoDB 업데이트
        logger.info("MongoDB 업데이트 시작...")
        update_mongodb(collection, index_info)
        logger.info("MongoDB 업데이트 완료")

    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()