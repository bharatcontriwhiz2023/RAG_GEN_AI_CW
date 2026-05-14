from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
data = PyPDFLoader("document_loaders/web.pdf")
docs = data.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)


template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI summarizer"), ("human", "{data}")]
)
model = ChatMistralAI(model="mistral-medium-3.5", temperature=0.2)
prompt = template.format_messages(data=docs)

response = model.invoke(prompt)

print(response.content)
