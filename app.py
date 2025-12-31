import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from src.ingestion import load_and_split_pdf
from src.rag_engine import setup_rag_chain

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")
    st.stop()

st.set_page_config(page_title="Smart Legal Reviewer", layout="wide")

st.title("⚖️ Smart Legal Contract Reviewer")
st.markdown("Upload a contract (PDF) and ask questions about clauses, penalties, or terms.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

with st.sidebar:
    st.header("📄 Upload Contract")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.session_state.document_name != uploaded_file.name:
            with st.spinner("Processing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    docs = load_and_split_pdf(tmp_path)
                    
                    if not docs:
                        st.error("Failed to extract text from PDF. Please ensure the PDF contains readable text.")
                        os.remove(tmp_path)
                        st.stop()
                    
                    st.session_state.rag_chain = setup_rag_chain(docs)
                    st.session_state.document_name = uploaded_file.name
                    
                    st.session_state.messages = []
                    
                    os.remove(tmp_path)
                    
                    st.success(f"✅ Contract '{uploaded_file.name}' processed! You can now ask questions.")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)
        else:
            st.info(f"📋 Currently analyzing: **{uploaded_file.name}**")
    
    if st.session_state.rag_chain:
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - What is the termination clause?
        - Are there any penalties for late delivery?
        - What are the payment terms?
        - Is there a non-compete clause?
        - What is the liability limit?
        """)
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if not st.session_state.rag_chain:
    st.info("👈 Please upload a PDF contract in the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the contract..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing contract..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})