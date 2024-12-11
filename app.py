import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings  # Use the updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize API Keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Streamlit App Title
st.title("RAG Document Q&A With Groq And Llama3")

# Initialize LLM and Prompt
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama-70b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    """
    Creates embeddings for the documents and initializes the FAISS vector database.
    """
    if "vectors" not in st.session_state:
        try:
            # Step 1: Check Directory
            if not os.path.exists("research_papers"):
                st.error("The directory 'research_papers' does not exist. Please add your PDF files.")
                return

            # Step 2: Load Documents
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")
            st.session_state.docs = st.session_state.loader.load()
            if not st.session_state.docs:
                st.error("No documents were loaded. Please check the 'research_papers' directory.")
                return
            st.write(f"{len(st.session_state.docs)} documents loaded.")

            # Step 3: Split Documents into Chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )
            st.write(f"{len(st.session_state.final_documents)} document chunks created.")

            # Step 4: Create Embeddings
            st.session_state.embeddings = OllamaEmbeddings()  # Updated import

            # Step 5: Initialize Vector Database
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
            st.success("Vector database initialized successfully.")

        except Exception as e:
            st.error(f"Error during vector embedding: {e}")

# User Input
user_prompt = st.text_input("Enter your query from the research paper:")

# Button to Initialize Document Embeddings
if st.button("Document Embedding"):
    create_vector_embedding()

# Process User Query
if user_prompt:
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.error("Vectors are not initialized. Click on 'Document Embedding' to create the vector database.")
    else:
        try:
            # Create Chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retriever_chain = create_retrieval_chain(retriever, document_chain)

            # Process Query
            start = time.process_time()
            response = retriever_chain.invoke({'input': user_prompt})
            st.write(f"Response time: {time.process_time() - start} seconds")
            if "answer" in response:
                st.write(f"**Answer:** {response['answer']}")
            else:
                st.write("No direct answer found. Here are the documents that may contain relevant information:")
                with st.expander("Document similarity search"):
                    if "context" in response:
                        for i, doc in enumerate(response['context']):
                            st.write(f"**Document {i+1}:**")
                            st.write(doc.page_content)
                            st.write("--------------------------")
                    else:
                        st.write("No context found for the response.")
        except Exception as e:
            st.error(f"Error during query processing: {e}")
