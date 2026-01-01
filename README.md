# ⚖️ Smart Legal Contract Reviewer

A Retrieval-Augmented Generation (RAG) application built with **LangChain** and **Streamlit** that helps users analyze legal contracts. It ingests PDF documents, creates vector embeddings, and uses an LLM (OpenAI GPT-3.5) to answer questions with high accuracy and context awareness.

---

## 🚀 Features

-   **Document Ingestion**: Upload PDF contracts directly via the UI.
-   **RAG Architecture**: Uses `RecursiveCharacterTextSplitter` to chunk complex legal text without losing context.
-   **Vector Search**: Implements **FAISS** for fast, local similarity search to retrieve relevant clauses.
-   **Interactive Chat**: Streamlit-based chat interface to query specific terms (e.g., "What is the termination penalty?").
-   **Source Context**: The retrieval engine identifies the specific sections of the PDF used to generate answers.

---

## 🛠️ Tech Stack

-   **Language**: Python 3.11
-   **Frameworks**: [LangChain](https://python.langchain.com/), [Streamlit](https://streamlit.io/)
-   **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
-   **LLM**: OpenAI GPT-3.5 Turbo
-   **PDF Processing**: `pypdf`

---

## 📂 Project Structure

```text
legal-rag-app/
├── .env                # API Keys (Not committed)
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
├── app.py              # Streamlit Frontend
└── src/                # RAG Logic
    ├── __init__.py
    ├── ingestion.py    # PDF loading & text splitting logic
    └── rag_engine.py   # Vector store setup & Retrieval chain
