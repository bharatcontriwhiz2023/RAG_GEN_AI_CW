from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="HTML is used for structuring web pages.",
        metadata={"source": "html.txt"},
    ),
    Document(
        page_content="CSS is used for styling web pages.",
        metadata={"source": "css.txt"},
    ),
    Document(
        page_content="JavaScript is used for adding interactivity to web pages.",
        metadata={"source": "javascript.txt"},
    ),
]

embeddings = MistralAIEmbeddings(model="mistral-embed")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db",
)

result = vectorstore.similarity_search("How we style web pages?", k=2)
for r in result:
    print(r.page_content)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("How we style web pages?")
