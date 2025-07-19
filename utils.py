from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

import tempfile
import os
import base64


# embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

llm = None  # global placeholder

def init_llm(groq_api_key):
    global llm
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.2,
    )

def save_file_to_disk(file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def show_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)


def create_vector_store(chunks):
    return FAISS.from_documents(chunks, embedding_model)


def retrieve_top_k_docs(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)


def query_with_chain(question, docs):
    if llm is None:
        raise ValueError("LLM is not initialized. Call init_llm() first.")
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=question)
