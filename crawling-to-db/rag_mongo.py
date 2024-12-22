from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from pymongo import MongoClient
import os
from langchain_openai import OpenAI
import logging
from langchain_core.runnables import RunnablePassthrough


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="RAG API")

# Pydantic 모델 정의
class QuestionRequest(BaseModel):
    question: str
    session_id: str

# API 키 및 환경변수 로드
load_dotenv()

# MongoDB 설정
try:
    # MongoDB 연결 설정
    mongo_path = os.getenv("MONGO_PATH")
    logger.info(f"Attempting to connect to MongoDB...")
    
    mongo_client = MongoClient(mongo_path)
    db = mongo_client["notice-db"]
    collection = db["final_notices"]
    
    # OpenAI Embeddings 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    logger.info("OpenAI embeddings initialized")
    
    # Vector Store 초기화
    vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            text_key="context",
            embedding_key="embedding",
            metadata_field_keys=["title", "department", "date", "link"],
            relevance_score_fn = "cosine"
        )
    logger.info("Vector store initialized")

except Exception as e:
    logger.error(f"Error during MongoDB setup: {str(e)}", exc_info=True)
    raise

# 프롬프트 템플릿 정의
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant answering questions about university notices.
You must respond in Korean language.

Instructions:
1. Answer based ONLY on the provided context or chat history
2. Be concise and direct
3. Include specific details (dates, numbers, names)
4. Always cite your source URL
5. Response must be in Korean

Format:
[한국어로 된 간단명료한 답변과 구체적인 정보]

출처:
- [공지사항 URL]

If you can't find the answer in the context or chat history, say "해당 질문에 대한 충분한 정보가 없습니다."

Previous chat history: {chat_history}

Question: {question}
Context: {context}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("assistant", "{context}")
])

# LLM 설정
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    """
    RAG를 사용한 질문-답변 엔드포인트
    """
    try:
        # 세션 기록을 저장할 딕셔너리
        store = {}

        # 세션 ID를 기반으로 세션 기록을 가져오는 함수
        def get_session_history(session_id):
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]
        # 세션 기록 가져오기
        chat_history = get_session_history(request.session_id)
        messages = getattr(chat_history, 'messages', [])
        
        # 질문을 문자열로 변환
        question = str(request.question)
        
        # retriever 설정
        retriever = vector_store.as_retriever()
        
        # 체인 생성
        rag_chain = (
            {
                "context": itemgetter("question") | retriever,  # 질문으로 문서 검색
                "question": RunnablePassthrough(),  # 질문을 그대로 전달
                "chat_history": itemgetter("chat_history"),
            }
            | PROMPT_TEMPLATE
            | llm
        )
        
        # RunnableWithMessageHistory 설정
        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        
        # 응답 생성
        response = chain_with_history.invoke(
            {"question": question},  # 문자열로 변환된 질문 사용
            config={"configurable": {"session_id": request.session_id}},
        )
        
        # 대화 기록 업데이트
        chat_history.add_user_message(question)
        chat_history.add_ai_message(response.content)
        
        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)