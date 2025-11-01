from dotenv import load_dotenv
import os
from openai import OpenAI
from numpy import dot
from numpy.linalg import norm
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

base_text = "織田信長は1560年に桶狭間の戦いで今川義元を破った。"

base_emb = client.embeddings.create(
  model = "text-embedding-3-small",
  input = base_text
)
base_vec = base_emb.data[0].embedding

compare_texts = [
  "こーちは28歳の男性です",
  "おーい！お茶",
  "今日はいい天気ですね。",
  "徳川家康は1600年に関ヶ原の戦いで石田三成率いる西軍を倒した！",
  "今川義元は、織田信長に桶狭間で敗れた。",
  "織田信長って人は、1560年に今川義元を桶狭間で破ったんだよね。"
]

compare_embs = client.embeddings.create(
  model = "text-embedding-3-small",
  input = compare_texts
)
compare_vecs = [e.embedding for e in compare_embs.data]

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))  

for text, vec in zip(compare_texts, compare_vecs):
    sim = cosine_similarity(base_vec, vec)
    print(f"{text}: {sim:.4f}")