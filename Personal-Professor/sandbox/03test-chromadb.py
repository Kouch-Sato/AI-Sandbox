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
  "本日の天気は雨です",
  "傘を持っていかないと",
  "私は神だ！",
  "株式会社Flamers",
  "This is a pen."
]

collection.add(
  documents = texts, 
  ids = [f"doc_{i}" for i in range(len(texts))]
)

query_text = "私は誰？"
results = collection.query(
  query_texts = [query_text],
  n_results = 3
)   

print(results["documents"][0][0])
print(results["documents"][0][1])
print(results["documents"][0][2])
