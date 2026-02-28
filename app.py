import streamlit as st
import os
import warnings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
# Fixed for LangChain v1 (2026 Release)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Suppress minor warnings for a clean UI
warnings.filterwarnings("ignore")

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="KL University Assistant", page_icon="🤖")
st.title("🤖 KL University AI Assistant")
st.markdown("Ask me anything about university rules, attendance, or CGPA!")

# --- 2. CONFIGURATION & SECRETS ---
# This pulls your new Groq key securely from Streamlit Cloud Secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit Secrets!")
    st.stop()

# --- 3. LOAD DATA (Cached to be fast) ---
@st.cache_resource
def initialize_rag():
    # Load the knowledge base
    if not os.path.exists("knowledge_base.txt"):
        st.error("Missing 'knowledge_base.txt' file!")
        return None

    try:
        loader = TextLoader("knowledge_base.txt", encoding="utf-8")
        docs = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Create vector database
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # --- FIXED SECTION: INITIALIZE GROQ LLM ---
        # We define the LLM here so it is part of the cached RAG chain
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        
        # System prompt for the University context
        system_prompt = (
            "You are an intelligent assistant for KL University. "
            "Use the following pieces of retrieved context to answer the question. "
            "If the answer is not in the context, say 'I don't have that information in my data.' "
            "\n\nContext: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Build the RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)
        
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

# Start the RAG engine
rag_chain = initialize_rag()

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Box
if user_input := st.chat_input("Type your question here..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        if rag_chain:
            try:
                with st.spinner("Consulting KL Knowledge Base..."):
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Chat Error: {e}")
        else:
            st.error("System is offline. Please check your API key.")
