from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document_loaders/text.pdf")

docs = data.load()

print(len(docs))
