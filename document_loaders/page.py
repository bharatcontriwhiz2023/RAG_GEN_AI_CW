from langchain_community.document_loaders import WebBaseLoader

url = "https://contriwhiz.com/company/"
data = WebBaseLoader(url)

docs = data.load()
print(docs)
