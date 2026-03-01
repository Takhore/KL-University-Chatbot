import os
import warnings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCtuTnrpJWTXfPBIItNbvoSA_SciIyvOGk" 

def main():
    print("Loading document and building vector database... Please wait.")
    
    # 1. Load the Data
    if not os.path.exists("knowledge_base.txt"):
        print("Error: 'knowledge_base.txt' not found!")
        return

    loader = TextLoader("knowledge_base.txt", encoding="utf-8")
    docs = loader.load()

    # 2. Split the Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Create Embeddings & Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Set Up the LLM (Removed the undefined variable call)
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    
    system_prompt = (
        "You are an intelligent assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, say 'I don't have that information in my data.' "
        "Keep the answer concise and strictly based on the provided data."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n" + "="*50)
    print("🤖 RAG Chatbot is ready! Ask questions about your data.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            response = rag_chain.invoke({"input": user_input})
            print(f"\nBot: {response['answer']}\n")
        except Exception as e:
            print(f"\nBot: Encountered an error - {e}\n")

if __name__ == "__main__":
    main()