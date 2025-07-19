import streamlit as st
import os
from dotenv import load_dotenv
from utils import init_llm, save_file_to_disk, load_file, split_docs, create_vector_store, retrieve_top_k_docs, query_with_chain

# Set page config
st.set_page_config(page_title="📄 LangChain RAG Chatbot", layout="wide")

# Language toggle
language = st.radio("🌐 Select Language | اختر اللغة", ["English", "العربية"], horizontal=True)

# RTL CSS if Arabic selected
if language == "العربية":
    st.markdown(
        """
        <style>
            body {
                direction: RTL;
                text-align: right;
            }
            .stTextInput > div > input {
                text-align: right;
            }
            .stMarkdown {
                direction: RTL;
                text-align: right;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Language dictionaries
text = {
    "English": {
        "title": "🧠 RAG Assistant with LangChain + Groq",
        "upload": "Upload files (PDF, TXT, DOCX)",
        "loading": "Loading & processing files...",
        "uploaded": "Uploaded and indexed {} document(s).",
        "ask": "Ask a question based on your documents:",
        "retrieving": "Retrieving answer...",
        "answer": "📌 Answer",
        "chunks": "📚 Retrieved Chunks"
    },
    "العربية": {
        "title": "🧠 مساعد الذكاء الاصطناعي باستخدام LangChain و Groq",
        "upload": "تحميل الملفات (PDF, TXT, DOCX)",
        "loading": "جارٍ تحميل ومعالجة الملفات...",
        "uploaded": "تم تحميل وفهرسة {} ملف(ات).",
        "ask": "اكتب سؤالك بناءً على الملفات:",
        "retrieving": "جارٍ استرجاع الإجابة...",
        "answer": "📌 الإجابة",
        "chunks": "📚 المقاطع المسترجعة"
    }
}

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("Groq API key not found in environment or Streamlit secrets!")
        st.stop()

# LLM
init_llm(GROQ_API_KEY)

st.title(text[language]["title"])

# File upload
uploaded_files = st.file_uploader(text[language]["upload"], type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

if uploaded_files:
    all_docs = []

    with st.spinner(text[language]["loading"]):
        for file in uploaded_files:
            path = save_file_to_disk(file)
            docs = load_file(path)
            all_docs.extend(docs)

        chunks = split_docs(all_docs)
        vectorstore = create_vector_store(chunks)

    st.success(text[language]["uploaded"].format(len(all_docs)))

    st.markdown("---")
    user_query = st.text_input(text[language]["ask"])

    if user_query:
        
        final_query = f"أجب على هذا السؤال باللغة العربية فقط: {user_query}" if language == "العربية" else user_query
        with st.spinner(text[language]["retrieving"]):
            retrieved_docs = retrieve_top_k_docs(vectorstore, user_query)
            answer = query_with_chain(final_query, retrieved_docs)

        st.subheader(text[language]["answer"])
        st.write(answer)

        with st.expander(text[language]["chunks"]):
            for doc in retrieved_docs:
                st.markdown(f"> {doc.page_content}")
