import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from pymongo import MongoClient
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="MongoDB RAG Chat Interface",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# MongoDB 및 RAG 초기화
@st.cache_resource
def initialize_rag():
    load_dotenv()
    
    try:
        # MongoDB 연결 설정
        mongo_path = os.getenv("MONGO_PATH")
        logger.info("Attempting to connect to MongoDB...")
        
        mongo_client = MongoClient(mongo_path)
        db = mongo_client["notice-db"]
        collection = db["test_embedded"]
        
        # MongoDB 연결 테스트 및 데이터 확인
        try:
            mongo_client.admin.command('ping')
            doc_count = collection.count_documents({})
            logger.info(f"MongoDB 연결 성공! 전체 문서 수: {doc_count}")
            
            # 샘플 문서 확인
            sample_doc = collection.find_one()
            if sample_doc:
                logger.info(f"샘플 문서 필드: {list(sample_doc.keys())}")
            else:
                logger.warning("컬렉션에 문서가 없습니다!")
        except Exception as e:
            logger.error(f"MongoDB 연결/데이터 확인 중 오류: {str(e)}")
            raise
        
        # OpenAI Embeddings 초기화
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        logger.info("OpenAI embeddings initialized")
        
        # Vector Store 초기화
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            text_key="context"
        )
        logger.info("Vector store initialized")
        
        # Retriever 설정 및 테스트
        retriever = vector_store.as_retriever()
        logger.info("Retriever 설정 완료")
        
        # 검색 테스트
        try:
            test_results = retriever.get_relevant_documents("test")
            logger.info(f"검색 테스트 결과: {len(test_results)} 문서 검색됨")
        except Exception as e:
            logger.error(f"검색 테스트 중 오류: {str(e)}")
        
        # 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. 
Our primary mission is to answer questions based on provided context or chat history.
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
Your final answer here, with numerical values, technical terms, jargon, and names in their original language
**Source**(Optional)
(Source of the answer, must be a file name(with a page number) or URL from the context. Omit if the answer is based on previous conversation or can't find the source.)
(list more if there are multiple sources)
...
###
Remember:
It's crucial to base your answer solely on the **provided context** or **chat history**. 
DO NOT use any external knowledge or information not present in the given materials.
If a user asks based on the previous conversation, but if there's no previous conversation or not enough information, you should answer that you don't know.
###
# Here is the user's question:
{question}
# Here is the context that you should use to answer the question:
{context}
# Your final answer to the user's question:"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            ("assistant", "{context}")
        ])
        
        # LLM 설정
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        
        # RAG 체인 생성
        rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | llm
        )
        
        return rag_chain
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        raise

# UI 구성
st.title("MongoDB RAG Chat Interface 🤖")
st.markdown("---")

# 채팅 히스토리 표시
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력
prompt = st.chat_input("질문을 입력하세요...")
if prompt:
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # RAG 체인 실행
        rag_chain = initialize_rag()
        
        # 세션 기록 관리
        store = {
            "default": st.session_state.chat_history
        }
        
        def get_session_history(session_id):
            return store[session_id]
        
        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        
        # 응답 생성 및 표시
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                response = chain_with_history.invoke(
                    {"question": prompt},
                    config={"configurable": {"session_id": "default"}},
                )
                st.markdown(response.content)
        
        # 응답 저장
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")

# 사이드바 설정
with st.sidebar:
    st.title("설정")
    if st.button("대화 내역 초기화"):
        st.session_state.messages = []
        st.session_state.chat_history = ChatMessageHistory()
        st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("### 시스템 정보")
    st.info("MongoDB RAG 시스템이 연결되어 있습니다.")

if __name__ == "__main__":
    # 앱이 처음 로드될 때 RAG 시스템 초기화
    initialize_rag()