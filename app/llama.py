__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

prompt = hub.pull("rlm/rag-prompt-mistral")

# Load the embedding function
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Initialize Vector Store and use it as retriever
db = Chroma("oscar_docs", embedding_function=embedding_function, persist_directory="db")
retriever = db.as_retriever(search_kwargs={"k": 3})

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="result",
)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 100  # Change this value based on your model and your GPU VRAM pool.
n_batch = 4000  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 8000  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="../llama.cpp/models/mistral-7b-instruct-v0.1.Q8_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    temperature=0.2,
)

retrieval_chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    retriever=retriever,
    llm=llm,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    memory=memory,
)
