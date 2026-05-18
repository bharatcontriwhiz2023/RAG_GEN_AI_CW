# load website data
# split into chunks
# create embeddings
# store in chroma db

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Website URLs
urls = [
    "https://contriwhiz.com/",
    "https://contriwhiz.com/company/",
    "https://contriwhiz.com/career/",
    "https://contriwhiz.com/contact/",
    "https://contriwhiz.com/browser-extensions/",
    "https://contriwhiz.com/website-development/",
    "https://contriwhiz.com/artificial-intelligence-development/",
    "https://contriwhiz.com/ecommerce/",
    "https://contriwhiz.com/saas-development/",
    "https://contriwhiz.com/enterprise-software-development/",
    "https://contriwhiz.com/website-migration-services/",
    "https://contriwhiz.com/application-and-plugins-development/",
]

all_docs = []

# Load all website pages
for url in urls:
    print(f"Loading: {url}")

    loader = WebBaseLoader(url)

    docs = loader.load()

    all_docs.extend(docs)

print(f"\nTotal documents loaded: {len(all_docs)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(all_docs)

print(f"Total chunks created: {len(chunks)}")

# Create embeddings
embeddings = MistralAIEmbeddings(model="mistral-embed")

# Store into ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)

print("\nChromaDB created successfully.")
