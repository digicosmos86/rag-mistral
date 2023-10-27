__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# client = chromadb.HttpClient(
#     host="http://yxu150.ccv.brown.edu",
#     port=8000,
#     settings=Settings(
#         chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
#         chroma_client_auth_credentials=os.environ.get("CHROMA_TOKEN"),
#     )
# )
client = chromadb.PersistentClient(path="db")

client.heartbeat()  # this should work with or without authentication - it is a public endpoint
client.get_version()  # this should work with or without authentication - it is a public endpoint

collection = client.get_collection("oscar_docs")

print(collection.count())
