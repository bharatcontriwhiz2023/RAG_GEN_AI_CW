from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
embeddings_model = MistralAIEmbeddings(model="mistral-embed")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings_model)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
)
llm = ChatMistralAI(model="mistral-medium-3.5")
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant. Use only the provided context to answer the question. If you don't know the answer, just say that you don't know.""",
        ),
        (
            "human",
            """Context: {context} Question: {question}""",
        ),
    ]
)
print("RAG system is ready.")
print("Enter 0 to exit.")
while True:
    query = input("Enter your question: ")
    if query == "0":
        print("Exiting...")
        break
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    final_prompt = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    print("Answer from RAG AI :", response.content)
