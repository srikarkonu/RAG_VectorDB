import os
from dotenv import load_dotenv
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from click import prompt
from pypdf import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64

load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
print(OPENAI_API_KEY)  # Optional: to verify the key is loaded correctly

st.set_page_config(page_title="KONU")

# Encode the image
with open("D:/RAG Model/Konu logo.png", "rb") as img_file:
    encoded_img = base64.b64encode(img_file.read()).decode()

# Add image in markdown
st.markdown(
    f"""
    <div>
        <img src="data:image/png;base64,{encoded_img}" alt="KONU Logo" width="200">
    </div>
    <div style="text-align: center;">   
        <h1 style="margin-top: 10px;">KONU BUJJI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index copy")

def load_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.load_local("faiss_index copy", embeddings=embeddings, allow_dangerous_deserialization=True)
    return vector_store

def get_response_from_documents(question):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(question)
    prompt_template = """
    Use the following context to answer the question accurately. If the answer is not available, respond with "The answer is not available in the context."
    
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain({"input_documents": docs, "question": question})["output_text"]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

# Sidebar for PDF upload
with st.sidebar:
    st.title("📂 Upload PDFs")
    pdf_files = st.file_uploader("Upload PDF files:", accept_multiple_files=True)
    if st.button("Process Documents"):
        if pdf_files:
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.uploaded = True
                st.success("Documents processed successfully!")

# Main chat interface
st.title("💬 KONU Chatbot")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if st.session_state.uploaded:
        response = get_response_from_documents(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)