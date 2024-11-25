import streamlit as st
import tiktoken
from loguru import logger
import requests
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.document_loaders import PyMuPDFLoader  # PDF 관련 로더만 남김

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

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
    return documents

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(_text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(_text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
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

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        file_options = {
            "공사사규.pdf": "https://github.com/ahn-su-bok/241113_chatbot/raw/main/김포도시관리공사사규_2024.06.27.pdf",
            "공사지침.pdf": "https://github.com/ahn-su-bok/241113_chatbot/raw/main/김포도시관리공사 지침_2024.10.18..pdf",
            "예산편성기준.pdf": "https://github.com/ahn-su-bok/241113_chatbot/raw/main/2025년도 지방공기업 예산편성기준.pdf",
        }
        selected_file = st.selectbox("Choose a file", list(file_options.keys()))
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        file_url = file_options[selected_file]

        # GitHub에서 PDF 파일 로드
        files_text = get_text_from_github(file_url)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        if vectorstore is None:
            st.error("Failed to create vector store. Please check the input text chunks and embeddings.")
            return

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                         "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

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
                            st.markdown(doc.metadata['source'], help=doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

if __name__ == '__main__':
    main()

