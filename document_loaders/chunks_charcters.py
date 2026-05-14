from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

text_split = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1,
)

data = TextLoader("document_loaders/file2.txt", encoding="utf-8")

docs = data.load()

chunks = text_split.split_documents(docs)

print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()
    print()
