import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from streamlit_mic_recorder import mic_recorder
import requests
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Set environment variables
pdfs_directory = os.getenv("PDFS_DIRECTORY", "chat-with-pdf/pdfs/")
ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
livekit_api_url = os.getenv("LIVEKIT_API_URL")
livekit_api_key = os.getenv("LIVEKIT_API_KEY")

# Ensure required environment variables are set
if not livekit_api_url or not livekit_api_key:
    st.error("LiveKit API URL or API Key is missing. Please check your .env file.")
    st.stop()

# Template for chatbot
template = """
You are a Reception assistant for a restaurant and need to answer questions related to the restaurant and book a call.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, say you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model=ollama_model)
vector_store = InMemoryVectorStore(embeddings)

# Initialize the LLM model
model = OllamaLLM(model=ollama_model)

# Function to upload PDF
def upload_pdf(file):
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory, exist_ok=True)
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

# Function to load PDF
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Function to split text
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Function to index documents
def index_docs(documents):
    vector_store.add_documents(documents)

# Function to retrieve documents
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Function to answer questions
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Function to convert audio to text using LiveKit API
def speech_to_text(audio_file):
    headers = {
        "Authorization": f"Bearer {livekit_api_key}",
    }
    files = {"file": open(audio_file, "rb")}
    response = requests.post(livekit_api_url, headers=headers, files=files)
    
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        st.error(f"LiveKit API failed: {response.status_code} - {response.text}")
        return ""

# Streamlit UI
st.title("Chat with PDF using Voice Input")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    file_path = upload_pdf(uploaded_file)
    documents = load_pdf(file_path)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)
    st.success("PDF uploaded and indexed successfully!")

    # Voice input
    st.write("🎤 Speak your question:")
    audio = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹ Stop Recording", key="recorder")

    if audio:
        # Save the audio to a file
        audio_file = "audio.wav"
        with open(audio_file, "wb") as f:
            f.write(audio['bytes'])

        # Convert audio to text using LiveKit
        question = speech_to_text(audio_file)

        if question:
            st.chat_message("user").write(question)
            related_documents = retrieve_docs(question)
            answer = answer_question(question, related_documents)
            st.chat_message("assistant").write(answer)
