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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter



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
    # 정보 검색을 위한 retriever 생성
    retriever = vectorstore.as_retriever()

    # 새로운 ChatPromptTemplate 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            (   
                "system",
                 """You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. 
Your primary mission is to answer questions based on provided context or chat history.
Ensure your response is concise and directly addresses the question without any additional narration.

###

You may consider the previous conversation history to answer the question.

# Here's the previous conversation history:
{chat_history}

###

Your final answer should be written concisely (but include important numerical values, technical terms, jargon, and names), followed by the source of the information.

# Steps

1. Carefully read and understand the context provided.
2. Identify the key information related to the question within the context.
3. Formulate a concise answer based on the relevant information.
4. Ensure your final answer directly addresses the question.
5. List the source of the answer in bullet points, which must be a file name (with a page number) or URL from the context. Omit if the answer is based on previous conversation or if the source cannot be found.

# Output Format:
[Your final answer here, with numerical values, technical terms, jargon, and names in their original language]

**Source**(Optional)
- (Source of the answer, must be a file name(with a page number) or URL from the context. Omit if the answer is based on previous conversation or can't find the source.)
- (list more if there are multiple sources)
- ...

###

Remember:
- It's crucial to base your answer solely on the **provided context** or **chat history**. 
- DO NOT use any external knowledge or information not present in the given materials.
- If a user asks based on the previous conversation, but if there's no previous conversation or not enough information, you should answer that you don't know.

###

# Here is the user's question:
{question}

# Here is the context that you should use to answer the question:
{context}

# Your final answer to the user's question:""",
            ),  # 기존 템플릿을 시스템 메시지로 사용
            MessagesPlaceholder(variable_name="chat_history"),  # 대화 기록 추가
            ("human", "{question}"),  # 질문
            ("assistant", "{context}")  # 컨텍스트
        ]
    )

    # LLM 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 체인 생성
    rag_chain = (
        {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
        | prompt
        | llm
    )
        # 세션 기록을 저장할 딕셔너리
    store = {}


    # 세션 ID를 기반으로 세션 기록을 가져오는 함수
    def get_session_history(session_ids):
        print(f"[대화 세션ID]: {session_ids}")
        if session_ids not in store:  # 세션 ID가 store에 없는 경우
            # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
            store[session_ids] = ChatMessageHistory()
        return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환
    
    chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",
    # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)
    # 질문에 대한 응답 스트리밍
    # answer = rag_chain.stream("문수웅장학금에 대해서 설명해줘")
    # stream_response(answer)
    
    print("채팅을 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    
    while True:
        # 사용자 입력 받기
        question = input("\n질문을 입력하세요: ").strip()
        
        # 종료 조건 확인
        if question.lower() in ['quit', 'exit', '종료']:
            print("채팅을 종료합니다.")
            break
            
        try:
            # 응답 생성
            response = chain_with_history.invoke(
                {"question": question},
                config={"configurable": {"session_id": "abc123"}},
            )
            
            # 응답 출력
            print("\n답변:")
            print(response.content)
            
        except Exception as e:
            print(f"\n오류가 발생했습니다: {str(e)}")
            print("다시 질문해 주세요.")
if __name__ == "__main__":
    main()

