import streamlit as st
import os
import warnings
import subprocess

# --- CORRECT IMPORTS FOR 2026 ---
from langchain_community.document_loaders import TextLoader
# Using the more stable PlaywrightURLLoader for Cloud environments
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="KL University Assistant", page_icon="🤖")
st.title("🤖 KL University AI Assistant")

# --- 2. CONFIGURATION & SECRETS ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit Secrets!")
    st.stop()

# --- 3. DYNAMIC RAG INITIALIZATION ---
@st.cache_resource
def initialize_rag(url=None):
    try:
        if url:
            # Step A: Install Browser Binaries (required for Streamlit Cloud)
            try:
                subprocess.run(["playwright", "install", "chromium"], check=True)
            except Exception as e:
                st.warning(f"Note: Browser install check: {e}")
            
            # Step B: Load using PlaywrightURLLoader (better for CSR sites)
            loader = PlaywrightURLLoader(urls=[url], remove_selectors=["header", "footer"])
            st.toast(f"Analyzing: {url}")
        else:
            if not os.path.exists("knowledge_base.txt"):
                st.error("Missing 'knowledge_base.txt' file!")
                return None
            loader = TextLoader("knowledge_base.txt", encoding="utf-8")
        
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        
        system_prompt = (
            "You are an intelligent university assistant. "
            "Use the provided context to answer the question accurately. "
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)
        
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🌐 Knowledge Source")
    web_url = st.text_input("Enter Website URL:", placeholder="https://kluone.netlify.app/...")
    
    if st.button("Learn from Website"):
        if web_url:
            st.session_state.rag_chain = initialize_rag(url=web_url)
            st.success("Knowledge Base Updated!")
        else:
            st.warning("Please enter a URL.")
    
    if st.button("Reset to File"):
        st.session_state.rag_chain = initialize_rag(url=None)
        st.info("Reverted to knowledge_base.txt")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = initialize_rag()

rag_chain = st.session_state.rag_chain

# --- 5. CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if rag_chain:
            try:
                with st.spinner("Searching..."):
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Chat Error: {e}")
