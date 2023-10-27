__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from pathlib import Path

from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma

oscar_docs_path = Path("../oscar-documentation")

# Load the embedding function
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Create the vector store
db = Chroma(
    "oscar_docs", embedding_function=embedding_function, persist_directory="db"
)

for doc in oscar_docs_path.rglob("*.md"):
    # Load the document
    loader = UnstructuredMarkdownLoader(
        str(doc),
    )
    doc = loader.load()

    splitter = MarkdownTextSplitter()
    split_docs = splitter.split_documents(doc)

    db.add_documents(split_docs)

db.persist()

query = "Who has access to the OSCAR cluster?"
result = db.similarity_search(query)

print(result[0].page_content)
