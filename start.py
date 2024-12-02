import streamlit as st
import tiktoken
from loguru import logger
import requests
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.document_loaders import PyMuPDFLoader 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate  

def download_file_from_github(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded file from {url}")
        return True
    else:
        logger.error(f"Failed to download file from {url}. Status code: {response.status_code}")
        return False

def get_text_from_github(file_url):
    local_path = "downloaded_file.pdf"
    success = download_file_from_github(file_url, local_path)
    
    if not success:
        raise ValueError(f"Failed to download file from {file_url}")

    if not os.path.exists(local_path):
        raise ValueError(f"File path {local_path} is not a valid file or url")

    loader = PyMuPDFLoader(local_path)  # PDF 로더만 사용
    documents = loader.load_and_split()
    
    for i, doc in enumerate(documents):
        doc.metadata['page'] = i + 1

    
    return documents

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(_text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(_text_chunks, embedding=embeddings)

    def compute_embeddings(data, model):
        embeddings = model.encode(data)
        return embeddings

    embeddings_cache = {}

    def cache_embeddings(data, model):
        for item in data:
            if item not in embeddings_cache:
                embeddings_cache[item] = compute_embeddings([item], model)
        return embeddings_cache

    return vector_store

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o', temperature=0)

    # PromptTemplate 생성
    template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    You must include 'page' number in your answer. 
    Answer in Korean.

    주어진 문서에서 다음 질문에 대한 답을 찾아주세요:
    {question}
    """
    prompt = PromptTemplate(input_variables=["question"], template=template)
 # stoutputparser 생성
    output_parser=StrOutputParser  

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain


def main():
    st.set_page_config(page_title="DirChat", page_icon=":books:")
    st.title("_김포도시관리공사 :red[QA Chat]_ :books:")

    # 초기 상태 설정
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
        
    # 환경 변수에서 API 키 가져오기 
    openai_api_key = os.getenv('OPENAI_API_KEY') 

    # 비밀번호 입력
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("Enter password", type="password")
        if st.button("Login"):
            if password == 'guc1234':
                st.session_state.authenticated = True
                st.success("Password correct. You are now authenticated.")
            else:
                st.error("Incorrect password. Please try again.")
        return
       

    with st.sidebar:
        file_options = {
            "공사사규.pdf": "https://github.com/ahn-su-bok/241113_chatbot/raw/main/김포도시관리공사사규_2024.6.27(압축).pdf",
            "공사지침.pdf": "https://github.com/ahn-su-bok/241113_chatbot/raw/main/김포도시관리공사 지침_10.18(압축).pdf",
            "예산편성기준.pdf": "https://github.com/ahn-su-bok/241113_chatbot/raw/main/2025년도 지방공기업 예산편성기준.pdf",
        }
        selected_file = st.selectbox("Choose a file", list(file_options.keys()))
        process = st.button("Process")

        st.header("파일 및 API 선택 후 PORCESS 버튼을 눌러주세요.")
        st.subheader("공사 챗봇 구성")
        st.text("1. 사규 : 기획, 총무, 인사, 재무, 감사")
        st.text("2. 지침")
        st.text("1) 감사실")
        st.caption("청렴행동수칙, 청렴시민감사관, 업무추진비 집행, 취업제한 및 업무취급제한 사무, 비위면직자등의 취업제한사무, 인권경영 이행, 개인정보보호, 에너지절약추진위원회, 위험성평가, 재해발생 시 보고 및 대응")
        st.text("2) 기획조정실")
        st.caption("예산운용, 시민참여예산, ESG 경영자문위원회, 위원회 심의수당 지급, 고객서비스헌장, 직장 내 괴롭힘 예방 및 대응, 공무국외여행, 성희롱/성폭력 예방, 선택적복지제도, 복무관리 및 운영, 물품 정수관리, 정보공개 심의회, 규제입증위원회, 휴직자 복무관리, 임금피크제")
        st.text("3. 2025년 예산편성기준")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        file_url = file_options[selected_file]

        # GitHub에서 PDF 파일 로드
        files_text = get_text_from_github(file_url)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks, openai_api_key)
        if vectorstore is None:
            st.error("Failed to create vector store. Please check the input text chunks and embeddings.")
            return

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                         "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    if st.button("대화 데이터 초기화"):
        st.session_state.conversation = None
        st.session_state.chat_history = None
        st.session_state.processComplete = None
        st.session_state.messages = [{"role": "assistant", "content": "대화 데이터가 초기화되었습니다. 새로 시작하시려면 질문을 입력해주세요!"}]
        st.success("대화 데이터가 초기화되었습니다.")

    # 사용자 메시지 표시 및 입력 처리
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            if chain is None:
                st.error("챗봇이 초기화되지 않았습니다. 파일을 업로드하고 처리 과정을 완료해주세요.")
            else:
                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                      for doc in source_documents:
                        page_number = doc.metadata.get('page', '페이지 정보 없음')
                        st.markdown(f"페이지 번호: {page_number}\n출처: {doc.metadata['source']}", help=doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": response})
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

if __name__ == '__main__':
    main()

