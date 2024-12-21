from byaldi import RAGMultiModalModel
from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from byaldi.colpali import ColPaliModel
from byaldi.objects import Result

# Optional langchain integration
try:
    from byaldi.integrations import ByaldiLangChainRetriever
except ImportError:
    pass

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        device: str = "cuda",
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
        instance.model = ColPaliModel.from_pretrained(
            pretrained_model_name_or_path,
            index_root=index_root,
            device=device,
            verbose=verbose,
        )
        return instance

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        index_root: str = ".byaldi",
        device: str = "cuda",
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
        index_path = Path(index_path)
        instance.model = ColPaliModel.from_index(
            index_path, index_root=index_root, device=device, verbose=verbose
        )

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

class MultiModalRAG:
    _instance = None  # 싱글톤 패턴을 위한 클래스 변수
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 환경 변수 로드
            load_dotenv()
            
            # 환경 변수 설정
            os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            
            # RAG 모델 초기화 (한 번만 실행됨)
            logger.info("RAG 모델 초기화 중...")
            cls._instance.model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)
            
        return cls._instance

    def create_index(self, images: List[Image.Image], metadata: List[Dict] = None):
        """이미지 리스트로부터 인덱스 생성"""
        try:
            logger.info("인덱싱 시작...")
            
            # 인덱스 생성
            for idx, image in enumerate(images):
                image_metadata = metadata[idx] if metadata else {"image_id": str(idx)}
                self.model.add_to_index(
                    input_item=image,
                    store_collection_with_index=True,
                    doc_id=idx,
                    metadata=image_metadata
                )
            
            logger.info("인덱싱 완료")
            return self.model.get_doc_ids_to_file_names()
            
        except Exception as e:
            logger.error(f"인덱싱 중 오류 발생: {str(e)}")
            raise
    
def main():
    try:
        # 1. RAG 모델 초기화
        rag = MultiModalRAG()
        
        # 2. 이미지 경로 설정 (실제 이미지 경로로 수정 필요)
        image_paths = [
            "path/to/image1.jpg",
            "path/to/image2.jpg"
        ]
        
        # 3. 이미지와 메타데이터 준비
        images = []
        metadata = []
        for idx, path in enumerate(image_paths):
            try:
                image = Image.open(path)
                images.append(image)
                metadata.append({
                    "image_id": str(idx),
                    "filename": Path(path).name
                })
            except Exception as e:
                logger.error(f"이미지 로드 중 오류 발생 ({path}): {str(e)}")
                continue
        
        if not images:
            logger.error("처리할 이미지가 없습니다.")
            return
        
        # 4. 인덱스 생성 및 결과 반환
        logger.info("인덱싱 시작...")
        index_info = rag.create_index(images, metadata)
        
        logger.info("인덱싱 완료")
        logger.info(f"생성된 인덱스 정보: {index_info}")
        
        return index_info
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    