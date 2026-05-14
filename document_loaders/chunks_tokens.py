from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("document_loaders/text.pdf")

docs = data.load()
text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=10)
chunks = text_splitter.split_documents(docs)

print(chunks[0].page_content)
