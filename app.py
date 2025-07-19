import streamlit as st
import os
from dotenv import load_dotenv
from utils import init_llm, save_file_to_disk, load_file, split_docs, create_vector_store, retrieve_top_k_docs, query_with_chain

st.set_page_config(page_title="📄 LangChain RAG Chatbot", layout="wide")

load_dotenv()
# getting Groq API key from Streamlit secrets or environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("Groq API key not found in environment or Streamlit secrets!")
        st.stop()

# LLM with the API key
init_llm(GROQ_API_KEY)

st.title("🧠 RAG Assistant with LangChain + Groq")

uploaded_files = st.file_uploader("Upload files (PDF, TXT, DOCX)", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

if uploaded_files:
    all_docs = []

    with st.spinner("Loading & processing files..."):
        for file in uploaded_files:
            path = save_file_to_disk(file)
            docs = load_file(path)
            all_docs.extend(docs)
        
        
        chunks = split_docs(all_docs)
        vectorstore = create_vector_store(chunks)

    st.success(f"Uploaded and indexed {len(all_docs)} document(s).")

    st.markdown("---")
    user_query = st.text_input("Ask a question based on your documents:")

    if user_query:
        with st.spinner("Retrieving answer..."):
            retrieved_docs = retrieve_top_k_docs(vectorstore, user_query)
            answer = query_with_chain(user_query, retrieved_docs)

        st.subheader("📌 Answer")
        st.write(answer)

        with st.expander("📚 Retrieved Chunks"):
            for doc in retrieved_docs:
                st.markdown(f"> {doc.page_content}")
