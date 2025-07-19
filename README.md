# AI powered document assistant (LangChain RAG Chatbot with Groq)

A Streamlit app that lets you upload PDFs, DOCX, and TXT files, indexes their content using embeddings, and answers questions using a Retrieval-Augmented Generation (RAG) approach powered by LangChain and Groq's LLM.

---

## Features

- Upload multiple documents (PDF, DOCX, TXT).
- Extract text, chunk documents for better retrieval.
- Embed chunks using HuggingFace embeddings (`BAAI/bge-small-en-v1.5`).
- Index chunks with FAISS for fast similarity search.
- Use Groq’s powerful `llama3-70b-8192` model for question answering.
- Interactive Streamlit UI for uploading files and querying.

---

## Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/ahmad786writes/AI-powered-document-assistant.git
   cd AI-powered-document-assistant
   ```

2. **Create and activate a conda environment**

   ```bash
   conda create -n rag-chatbot python -y
   conda activate rag-chatbot
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   _Dependencies include: Streamlit, LangChain, LangChain Community loaders, HuggingFace transformers, FAISS, Groq SDK, python-dotenv._

4. **Get your Groq API key**

   - Sign up for Groq Cloud and get an API key.
   - Set the key in your environment or Streamlit secrets:

   ```bash
   export GROQ_API_KEY="your_api_key"  # Linux/macOS
   set GROQ_API_KEY="your_api_key"     # Windows
   ```

   Or in `.streamlit/secrets.toml`:

   ```toml
   GROQ_API_KEY = "your_api_key"
   ```

   Or create an .env file and save it with 
   ```bash	
   GROQ_API_KEY="your_api_key"
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## How It Works

1. **Upload documents**: You upload one or more PDF, DOCX, or TXT files.

2. **File processing**:
   - Files are saved temporarily.
   - Text is extracted using appropriate loaders (`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`).
   - Text is split into chunks (default 500 characters with 100 overlap) for efficient embedding and retrieval.

3. **Embedding & Indexing**:
   - Each chunk is embedded via `HuggingFaceEmbeddings` (model `BAAI/bge-small-en-v1.5`).
   - The chunks are indexed in a FAISS vector store for similarity search.

4. **Querying**:
   - When you ask a question, top relevant chunks are retrieved by similarity search.
   - Retrieved chunks are passed to the Groq LLM (`llama3-70b-8192`) with a question-answering chain.
   - The LLM generates a concise answer based on the retrieved context.

5. **Result**:
   - The answer is displayed in the UI.
   - You can optionally expand to see the chunks used to generate the answer.

---

## File Structure

- `app.py`: Main Streamlit app interface.
- `utils.py`: Helper functions for file saving, loading, splitting, embedding, vector store creation, and querying.
- `.env`: (optional) Environment variables like API keys.

---

## Notes

- Supported file types: PDF, DOCX, TXT.
- Groq API key is mandatory for the LLM to initialize.
- FAISS index is rebuilt on each upload session.
- Text splitter helps LLM handle long documents by chunking content smartly.
- The embedding model is lightweight for efficient vectorization.

---

## Troubleshooting

- **Groq API key missing or invalid**: Make sure the key is set in environment variables or Streamlit secrets.
- **Unsupported file type**: Upload only PDF, DOCX, or TXT files.
- **Slow responses**: Large files or many chunks can slow down embedding and indexing.
- **Streamlit errors**: Ensure all dependencies are installed and your Python version is compatible (recommended 3.8+).

---

Built with ❤️ by Ahmad Liaqat — AI Engineer focused on making RAG easy and practical.

---

Feel free to ask questions or suggest features!
