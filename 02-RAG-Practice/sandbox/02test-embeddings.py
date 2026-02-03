from dotenv import load_dotenv
import os
from openai import OpenAI
from numpy import dot
from numpy.linalg import norm
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = "text-embedding-3-large"

base_text = "今日は雨が降ってるな"

base_emb = client.embeddings.create(
  model = embedding_model,
  input = base_text
)
base_vec = base_emb.data[0].embedding

compare_texts = [
  "本日の天気は雨です",
  "傘を持っていかないと",
  "天気予報を確認しよう",
  "私は神だ！",
  "株式会社Flamers",
  "This is a pen."
]

compare_embs = client.embeddings.create(
  model = embedding_model,
  input = compare_texts
)
compare_vecs = [e.embedding for e in compare_embs.data]

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))  

for text, vec in zip(compare_texts, compare_vecs):
    sim = cosine_similarity(base_vec, vec)
    print(f"{text}: {sim:.4f}")