from dotenv import load_dotenv
import os
from openai import OpenAI
from numpy import dot
from numpy.linalg import norm
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
embedding_model = "text-embedding-3-large"

client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
  api_key = os.getenv("OPENAI_API_KEY"),
  model_name = embedding_model
)

collection = client.create_collection(
  name = "test_collection",
  embedding_function = openai_ef
)

texts = [
    "私はオランダで生まれて、東京で育ちました",
    "趣味はボードゲームとポーカーです。",
    "AIやスタートアップの世界にワクワクするタイプです。",
    "休日はカフェで読書することが多い。",
    "猫を飼っていて、毎朝一緒に起きます。",
    "なみしろさんという方と結婚しています"
]

collection.add(
  documents = texts, 
  ids = [f"doc_{i}" for i in range(len(texts))]
)

query_text = "趣味は何？"
results = collection.query(
  query_texts = [query_text],
  n_results = 6
)   

for i in range(6):
    print(results["documents"][0][i], results["distances"][0][i])
