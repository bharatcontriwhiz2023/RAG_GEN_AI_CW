from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
data = TextLoader("document_loaders/notes.txt", encoding="utf-8")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI summarizer"), ("human", "{data}")]
)
model = ChatMistralAI(model="mistral-medium-3.5", temperature=0.2)
propmt = template.format_messages(data=docs[0].page_content)

response = model.invoke(propmt)

print(response.content)
