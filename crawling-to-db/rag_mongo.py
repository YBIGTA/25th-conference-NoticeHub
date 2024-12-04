from dotenv import load_dotenv
from langchain.schema import Document
from pymongo.operations import SearchIndexModel
import time
import os
from pymongo import MongoClient
from langchain.schema import Document
from get_embedding import get_embedding


# MongoDB 설정
mongo_path = os.getenv("MONGO_PATH")
mongo_client = MongoClient(mongo_path)
db = mongo_client["notice-db"]
collection = db["test_embedded"]

def create_vector_search_index(collection, index_name="vector_index", batch_size=10):
    """
    MongoDB Atlas에 벡터 검색 인덱스를 배치로 생성하고 동기화를 확인하는 함수.

    Args:
        collection: MongoDB 컬렉션 객체.
        index_name (str): 생성할 인덱스의 이름 (기본값: "vector_index").
        batch_size (int): 한 번에 처리할 배치 크기 (기본값: 1000).

    Returns:
        bool: 모든 배치의 인덱스 생성 및 동기화 성공 여부.
    """
    try:
        # 총 문서 수 확인
        total_documents = collection.count_documents({})
        print(f"Total documents in collection: {total_documents}")

        # 배치 처리
        for i in range(0, total_documents, batch_size):
            print(f"Processing batch {i} to {i + batch_size}...")

            # 배치에 해당하는 문서 가져오기
            batch = list(collection.find().skip(i).limit(batch_size))
            if not batch:
                print(f"No documents found in batch {i} to {i + batch_size}")
                continue

            # 인덱스 모델 생성
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": 768,  # OpenAI 임베딩은 1536 또는 768
                            "path": "embedding",
                            "similarity": "cosine"
                        }
                    ]
                },
                name=f"{index_name}_batch_{i // batch_size}",
                type="vectorSearch"
            )

            # 인덱스 생성
            print(f"Creating vector search index for batch {i} to {i + batch_size}...")
            collection.create_search_index(model=search_index_model)

            # 동기화 상태 확인
            print("Polling to check if the index is ready...")
            while True:
                indices = list(collection.list_search_indexes(f"{index_name}_batch_{i // batch_size}"))
                if len(indices) and indices[0].get("queryable"):
                    print(f"Index '{index_name}_batch_{i // batch_size}' is ready for querying.")
                    break
                time.sleep(5)

        print("All batches indexed successfully!")
        return True

    except Exception as e:
        print(f"Error creating search index: {e}")
        return False

def mongo_retriever(query_embedding, limit=5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",  # MongoDB Atlas의 KNN 인덱스 이름
                "queryVector": query_embedding,
                "path": "embedding",
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0, "title": 1, "content": 1  # 필요한 필드만 반환
            }
        }
    ]
    results = collection.aggregate(pipeline)
    return [doc for doc in results]

def create_retriever(query):
    # OpenAI로 쿼리를 벡터화
    query_embedding = get_embedding(query)

    # MongoDB에서 검색
    mongo_results = mongo_retriever(query_embedding)

    # 검색 결과를 LangChain의 Document 형식으로 변환
    documents = [
        Document(page_content=res["content"], metadata={"title": res["title"]})
        for res in mongo_results
    ]
    return documents

def main():
    load_dotenv()

    # MongoDB에 벡터 검색 인덱스 생성
    create_vector_search_index(collection)

    # MongoDB에서 검색
    query = "문수웅 장학금에 대해 알려줘"
    retriever_results = create_retriever(query)
    for doc in retriever_results:
        print(f"Title: {doc.metadata['title']}\nContent: {doc.page_content}")

if __name__ == "__main__":
    main()