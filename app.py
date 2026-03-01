import streamlit as st
import os
import warnings
# --- UPDATED IMPORT: Using Playwright for dynamic web scraping ---
from langchain_community.document_loaders import TextLoader, PlaywrightWebBaseLoader
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
        # Step 1: Decide which data to load
        if url:
            # --- UPDATED: Playwright handles JavaScript-heavy sites like the faculty portal ---
            loader = PlaywrightWebBaseLoader(url)
            st.toast(f"Launching Headless Browser for: {url}")
        else:
            # Fallback to local Knowledge Base
            if not os.path.exists("knowledge_base.txt"):
                st.error("Missing 'knowledge_base.txt' file!")
                return None
            loader = TextLoader("knowledge_base.txt", encoding="utf-8")
        
        # This will now wait for the page to render before loading
        docs = loader.load()
        
        # Step 2: Split, Embed, and Store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Chroma handles the vector storage
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Step 3: Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        
        system_prompt = (
            "You are an intelligent university assistant. "
            "Use the provided context to answer the question accurately. "
            "If the answer isn't there, say you don't have that info."
            "\n\nContext: {context}"
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

# --- 4. SIDEBAR FOR URL INPUT ---
with st.sidebar:
    st.header("🌐 Web Study Mode (Headless)")
    st.write("Using Playwright to study dynamic websites.")
    web_url = st.text_input("Enter Website URL:", placeholder="https://www.klu.ac.in/...")
    
    if st.button("Learn from Website"):
        if web_url:
            st.session_state.rag_chain = initialize_rag(url=web_url)
            st.success("Headless Crawl Complete!")
        else:
            st.warning("Please enter a URL first.")
    
    if st.button("Reset to KLU File"):
        st.session_state.rag_chain = initialize_rag(url=None)
        st.info("Reverted to knowledge_base.txt")

# Start/Get the RAG engine from session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = initialize_rag()

rag_chain = st.session_state.rag_chain

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if rag_chain:
            try:
                with st.spinner("Analyzing data via Headless Browser..."):
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Chat Error: {e}")
        else:
            st.error("System is offline.")
