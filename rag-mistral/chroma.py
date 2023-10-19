import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

client = chromadb.HttpClient(
    host="http://yxu150.ccv.brown.edu",
    port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        chroma_client_auth_credentials=os.environ.get("CHROMA_TOKEN"),
    )
)
client.heartbeat()  # this should work with or without authentication - it is a public endpoint

client.get_version()  # this should work with or without authentication - it is a public endpoint

print(client.list_collections())
