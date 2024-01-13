import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
    page_title="sysnovaGPT",
    page_icon=":supervillain:")

    st.title(":supervillain: sysnova :red[private] :violet[GPT]")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.secrets["openai_api_key"]
        process = st.button("Submit")

        # new 벡터와 메타데이터 생성 및 다운로드 코드
        if st.button("Generate and Download TSV Files"):
            if uploaded_files:
                documents = get_text(uploaded_files)  # 파일에서 텍스트 추출
                chunks = get_text_chunks(documents)  # 텍스트를 청크로 분할
                vectordb = get_vectorstore(chunks)  # 청크에서 벡터 생성

                # 벡터와 메타데이터 DataFrame으로 변환
                df_vectors = pd.DataFrame([doc.embedding for doc in vectordb.documents])
                df_metadata = pd.DataFrame([{'Title': doc.title, 'Text': doc.text[:100] + '...'} for doc in vectordb.documents])

                # TSV 형식으로 변환
                tsv_vectors = df_vectors.to_csv(sep='\t', index=False, header=False)
                tsv_metadata = df_metadata.to_csv(sep='\t', index=False, header=True)

                # Streamlit 다운로드 버튼 추가
                st.download_button(
                    label="Download Vectors TSV",
                    data=tsv_vectors,
                    file_name="vectors.tsv",
                    mime="text/tsv"
                )
                st.download_button(
                    label="Download Metadata TSV",
                    data=tsv_metadata,
                    file_name="metadata.tsv",
                    mime="text/tsv"
                )
            else:
                st.warning("파일을 먼저 업로드해주세요.")
        # new 여기까지
        
    if process:
        #if not openai_api_key:
            #st.info("Please add your OpenAI API key to continue.")
            #st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "무엇이든 물어보세요! :point_left: 좌측 sidebar 문서 upload 먼저!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 자료 및 내용 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help = doc.page_content)

                #with st.expander("참고 문서 확인"):
                    #st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    #st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    #st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
