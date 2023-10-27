from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import LLMChain
from lanarky import LangchainRouter

from llamacpp import llm, template

load_dotenv()
app = FastAPI()

llmchain = LLMChain.from_string(llm=llm, template="{question}")

langchain_router = LangchainRouter(
    langchain_url="/chat", langchain_object=llmchain, streaming_mode=0
)

app.include_router(langchain_router)
