# rag.py

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_teddynote.messages import stream_response
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss  # FAISS 라이브러리 임포트
import pickle


def load_api_keys():
    # API 키를 환경변수로 관리하기 위한 설정 파일
    load_dotenv()


def main():
    load_api_keys()
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # 벡터스토어 파일 로드
    vectorstore = FAISS.load_local(
        folder_path="faiss_db",
        index_name="faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
)

    # vectorstore = load_vectorstore('vectorstore2.faiss', 'docstore2.pkl')

    # 정보 검색을 위한 retriever 생성
    retriever = vectorstore.as_retriever()

    # LLM 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 체인 생성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | hub.pull("teddynote/rag-prompt-korean")  # 프롬프트 가져오기
        | llm
        | StrOutputParser()
    )

    # 질문에 대한 응답 스트리밍
    answer = rag_chain.stream("문수웅장학금에 대해서 설명해줘")
    stream_response(answer)

if __name__ == "__main__":
    main()