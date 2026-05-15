from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI summarizer"), ("human", "{data}")]
)
model = ChatMistralAI(model="mistral-medium-3.5", temperature=0.2)
