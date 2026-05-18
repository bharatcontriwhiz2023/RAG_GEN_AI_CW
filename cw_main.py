import streamlit as st
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Contriwhiz AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    .stChatMessage {
        border-radius: 15px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .user-msg {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 15px;
        color: white;
    }

    .assistant-msg {
        background-color: #111827;
        padding: 15px;
        border-radius: 15px;
        color: #f8fafc;
        border: 1px solid #334155;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #38bdf8;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    .stChatInputContainer {
        border-top: 1px solid #334155;
        background-color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- HEADER --------------------
st.markdown('<div class="title">🤖 Contriwhiz AI Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask and get to know more about Contriwhiz</div>',
    unsafe_allow_html=True,
)

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- INITIALIZE MODELS --------------------
embeddings_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings_model)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)

llm = ChatMistralAI(model="mistral-medium-3.5")

# -------------------- PROMPT TEMPLATE --------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer.

If the answer is not available in the context,
say: "I don't know based on the provided data."
""",
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

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- DISPLAY CHAT HISTORY --------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-msg">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-msg">{message["content"]}</div>',
                unsafe_allow_html=True,
            )

# -------------------- CHAT INPUT --------------------
query = st.chat_input("Ask your question...")

if query:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-msg">{query}</div>',
            unsafe_allow_html=True,
        )

    # Retrieve documents
    docs = retriever.invoke(query)

    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create final prompt
    final_prompt = prompt_template.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    # Generate response
    response = llm.invoke(final_prompt)

    answer = response.content

    # Add assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(
            f'<div class="assistant-msg">{answer}</div>',
            unsafe_allow_html=True,
        )
