# create_vectorstore.py

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore  # InMemoryDocstore 임포트
import faiss  # FAISS 라이브러리 임포트
import pandas as pd
import pickle  # pickle 임포트


class VectorStoreManager:
    def __init__(self, csv_file):
        self.combined_df = pd.read_csv(csv_file, encoding='utf-8')
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 임베딩 모델 초기화
        self.dimension_size = len(self.embeddings.embed_query("hello world"))  # 임베딩 차원 크기 계산

    def split_records_to_documents(self):
        # 각 레코드를 Document로 변환
        documents = []
        for idx, row in self.combined_df.iterrows():
            # context가 NaN인 경우 빈 문자열로 대체
            context = row['context'] if pd.notna(row['context']) else ""
            
            document = Document(
                page_content=context,  # context를 문서 내용으로 사용
                metadata={
                    "index": idx,
                    "title": row['title'],
                    "date": row['date'],
                    "link": row['link'],
                    "department": row['department']
                }
            )
            documents.append(document)
        return documents

    def create_vectorstore(self):
        # 레코드를 Document로 변환
        documents = self.split_records_to_documents()
        
        # FAISS 벡터 저장소 생성
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(self.dimension_size),  # 임베딩 차원 크기에 맞춰 인덱스 생성
            docstore=InMemoryDocstore(),  # InMemoryDocstore를 사용하여 메타데이터 저장
            index_to_docstore_id={},  # 초기화
        )
        
        # DB 생성
        self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)

        return self.vectorstore

    def save_vectorstore(self, file_path, docstore_path):
        # FAISS 인덱스를 파일로 저장
        if self.vectorstore is not None:
            faiss.write_index(self.vectorstore.index, file_path)
            print(f"벡터스토어가 '{file_path}'에 저장되었습니다.")
            
            # docstore 저장
            with open(docstore_path, 'wb') as f:
                pickle.dump(self.vectorstore.docstore, f) 
                print(f"Docstore가 '{docstore_path}'에 저장되었습니다.")
        else:
            print("벡터스토어가 생성되지 않았습니다.")


if __name__ == "__main__":
    load_dotenv() 
    csv_file_path = '../data/combined_notices.csv' 
    vector_store_manager = VectorStoreManager(csv_file_path)
    
    vectorstore = vector_store_manager.create_vectorstore()

    vectorstore.save_local(folder_path="faiss_db", index_name="faiss_index")
    