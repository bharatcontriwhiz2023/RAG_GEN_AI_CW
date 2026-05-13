from langchain_community.document_loaders import TextLoader

data = TextLoader("document_loaders/notes.txt", encoding="utf-8")

docs = data.load()

print(docs[0])
