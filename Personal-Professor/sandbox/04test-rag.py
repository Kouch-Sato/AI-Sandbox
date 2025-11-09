from dotenv import load_dotenv
import os
from openai import OpenAI
from numpy import dot
from numpy.linalg import norm
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
embedding_model = "text-embedding-3-large"
openai_client = OpenAI(
  api_key = os.getenv("OPENAI_API_KEY")
)

chromadb_client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
  api_key = os.getenv("OPENAI_API_KEY"),
  model_name = embedding_model
)

collection = chromadb_client.get_or_create_collection(
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

collection.upsert(
  documents = texts, 
  ids = [f"doc_{i}" for i in range(len(texts))]
)

def rag_answer(question: str):
  response = collection.query(
    query_texts = [question],
    n_results = 3,
  )   

  contexts = []
  for i in range(3):
    contexts.append(f"[{i+1}] {response['documents'][0][i]}")

  prompt = f"""
    以下の情報をもとに質問に答えてください。
    あなたは厳密なリサーチアシスタントです。
    以下の「参照メモ」に書かれている内容【だけ】を根拠に、日本語で簡潔に答えてください。
    参照に無い内容は推測せず、「手元のメモには情報がありません」と述べてください。
    回答の最後に、使った根拠の番号を [1] のように列挙してください。

    参考メモ: {contexts}
    質問: {question}
  """

  chat_response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "あなたは厳密なリサーチアシスタントです。"},
        {"role": "user", "content": prompt}
    ]
  )

  return chat_response.choices[0].message.content

print("質問を入力してください")
user_question = input("質問: ")
answer = rag_answer(user_question)
print("回答:")
print(answer)