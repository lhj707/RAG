{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install streamlit\n",
    "!pip install tiktoken\n",
    "!pip install pysqlite3-binary\n",
    "!pip install langchain\n",
    "!pip install chromadb\n",
    "!pip install unstructured\n",
    "!pip install sentence-transformers\n",
    "!pip install faiss-cpu\n",
    "!pip install tiktoken\n",
    "!pip install openai\n",
    "!pip install pypdf\n",
    "!pip install loguru\n",
    "!pip install docx2txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tiktoken"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from loguru import logger"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.chains import ConversationalRetrievalChain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.chat_models import ChatOpenAI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.document_loaders import PyPDFLoader\n",
    "from langchain.document_loaders import Docx2txtLoader\n",
    "from langchain.document_loaders import UnstructuredPowerPointLoader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "from langchain.embeddings import HuggingFaceEmbeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.memory import ConversationBufferMemory\n",
    "from langchain.vectorstores import FAISS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# from streamlit_chat import message\n",
    "from langchain.callbacks import get_openai_callback\n",
    "from langchain.memory import StreamlitChatMessageHistory"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    st.set_page_config(\n",
    "    page_title=\"DirChat\",\n",
    "    page_icon=\":books:\")\n",
    "\n",
    "    st.title(\"_Private Data :red[QA Chat]_ :books:\")\n",
    "\n",
    "    if \"conversation\" not in st.session_state:\n",
    "        st.session_state.conversation = None\n",
    "\n",
    "    if \"chat_history\" not in st.session_state:\n",
    "        st.session_state.chat_history = None\n",
    "\n",
    "    if \"processComplete\" not in st.session_state:\n",
    "        st.session_state.processComplete = None\n",
    "\n",
    "    with st.sidebar:\n",
    "        uploaded_files =  st.file_uploader(\"Upload your file\",type=['pdf','docx'],accept_multiple_files=True)\n",
    "        openai_api_key = st.text_input(\"OpenAI API Key\", key=\"chatbot_api_key\", type=\"password\")\n",
    "        process = st.button(\"Process\")\n",
    "    if process:\n",
    "        if not openai_api_key:\n",
    "            st.info(\"Please add your OpenAI API key to continue.\")\n",
    "            st.stop()\n",
    "        files_text = get_text(uploaded_files)\n",
    "        text_chunks = get_text_chunks(files_text)\n",
    "        vetorestore = get_vectorstore(text_chunks)\n",
    "     \n",
    "        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) \n",
    "\n",
    "        st.session_state.processComplete = True\n",
    "\n",
    "    if 'messages' not in st.session_state:\n",
    "        st.session_state['messages'] = [{\"role\": \"assistant\", \n",
    "                                        \"content\": \"안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!\"}]\n",
    "\n",
    "    for message in st.session_state.messages:\n",
    "        with st.chat_message(message[\"role\"]):\n",
    "            st.markdown(message[\"content\"])\n",
    "\n",
    "    history = StreamlitChatMessageHistory(key=\"chat_messages\")\n",
    "\n",
    "    # Chat logic\n",
    "    if query := st.chat_input(\"질문을 입력해주세요.\"):\n",
    "        st.session_state.messages.append({\"role\": \"user\", \"content\": query})\n",
    "\n",
    "        with st.chat_message(\"user\"):\n",
    "            st.markdown(query)\n",
    "\n",
    "        with st.chat_message(\"assistant\"):\n",
    "            chain = st.session_state.conversation\n",
    "\n",
    "            with st.spinner(\"Thinking...\"):\n",
    "                result = chain({\"question\": query})\n",
    "                with get_openai_callback() as cb:\n",
    "                    st.session_state.chat_history = result['chat_history']\n",
    "                response = result['answer']\n",
    "                source_documents = result['source_documents']\n",
    "\n",
    "                st.markdown(response)\n",
    "                with st.expander(\"참고 문서 확인\"):\n",
    "                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)\n",
    "                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)\n",
    "                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)\n",
    "                    \n",
    "\n",
    "\n",
    "# Add assistant message to chat history\n",
    "        st.session_state.messages.append({\"role\": \"assistant\", \"content\": response})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def tiktoken_len(text):\n",
    "    tokenizer = tiktoken.get_encoding(\"cl100k_base\")\n",
    "    tokens = tokenizer.encode(text)\n",
    "    return len(tokens)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_text(docs):\n",
    "\n",
    "    doc_list = []\n",
    "    \n",
    "    for doc in docs:\n",
    "        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용\n",
    "        with open(file_name, \"wb\") as file:  # 파일을 doc.name으로 저장\n",
    "            file.write(doc.getvalue())\n",
    "            logger.info(f\"Uploaded {file_name}\")\n",
    "        if '.pdf' in doc.name:\n",
    "            loader = PyPDFLoader(file_name)\n",
    "            documents = loader.load_and_split()\n",
    "        elif '.docx' in doc.name:\n",
    "            loader = Docx2txtLoader(file_name)\n",
    "            documents = loader.load_and_split()\n",
    "        elif '.pptx' in doc.name:\n",
    "            loader = UnstructuredPowerPointLoader(file_name)\n",
    "            documents = loader.load_and_split()\n",
    "\n",
    "        doc_list.extend(documents)\n",
    "    return doc_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_text_chunks(text):\n",
    "    text_splitter = RecursiveCharacterTextSplitter(\n",
    "        chunk_size=900,\n",
    "        chunk_overlap=100,\n",
    "        length_function=tiktoken_len\n",
    "    )\n",
    "    chunks = text_splitter.split_documents(text)\n",
    "    return chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_vectorstore(text_chunks):\n",
    "    embeddings = HuggingFaceEmbeddings(\n",
    "                                        model_name=\"jhgan/ko-sroberta-multitask\",\n",
    "                                        model_kwargs={'device': 'cpu'},\n",
    "                                        encode_kwargs={'normalize_embeddings': True}\n",
    "                                        )  \n",
    "    vectordb = FAISS.from_documents(text_chunks, embeddings)\n",
    "    return vectordb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_conversation_chain(vetorestore,openai_api_key):\n",
    "    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)\n",
    "    conversation_chain = ConversationalRetrievalChain.from_llm(\n",
    "            llm=llm, \n",
    "            chain_type=\"stuff\", \n",
    "            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), \n",
    "            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),\n",
    "            get_chat_history=lambda h: h,\n",
    "            return_source_documents=True,\n",
    "            verbose = True\n",
    "        )\n",
    "\n",
    "    return conversation_chain\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
