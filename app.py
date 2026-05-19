import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ---------------- LOAD ENV ----------------
load_dotenv()


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
)


# ---------------- TITLE ----------------
st.title("📚 RAG PDF Chatbot")
st.markdown("Upload a PDF book and ask questions from it.")


# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])


# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- PROCESS PDF ----------------
if uploaded_file is not None:

    with st.spinner("Processing PDF and creating embeddings..."):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

        chunks = text_splitter.split_documents(docs)

        # Embeddings
        embeddings = MistralAIEmbeddings(model="mistral-embed")

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db",
        )

        st.session_state.vectorstore = vectorstore

    st.success("PDF uploaded and embeddings created successfully!")


# ---------------- CHAT SECTION ----------------
query = st.chat_input("Ask a question from the uploaded PDF...")


if query:

    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
    else:

        # Display user message
        st.chat_message("user").write(query)

        # Retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10,
                "lambda_mult": 0.5,
            },
        )

        # Get relevant docs
        docs = retriever.invoke(query)

        context = "\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.
                    Use only the provided context to answer the question.
                    If you don't know the answer, just say that you don't know.""",
                ),
                (
                    "human",
                    """Context:
                    {context}

                    Question:
                    {question}
                    """,
                ),
            ]
        )

        final_prompt = prompt_template.invoke(
            {
                "context": context,
                "question": query,
            }
        )

        # LLM
        llm = ChatMistralAI(model="mistral-medium-3.5")

        with st.spinner("Generating answer..."):
            response = llm.invoke(final_prompt)

        # Display assistant response
        st.chat_message("assistant").write(response.content)
