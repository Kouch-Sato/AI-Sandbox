from dotenv import load_dotenv
import os
from openai import OpenAI
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
  "趣味は自転車とポーカーです。",
  "AIについて勉強しています。",
  "休日はカフェで作業することが多いよ。",
  "猫を飼っていて、まろんという名前です",
  "2年前に結婚しました",
  "旅行が好きで、先月はシリコンバレーに行きました",
  "英語の勉強をしてて、もっと話せるようになりたいな",
  "赤い色が好きです！",
  "この前自転車を盗まれて、新しい自転車を買いました。悲しい…",
  "最近夜更かししてるから早く寝ないとなあ",
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
    contexts.append(response['documents'][0][i])

  prompt = f"""
    以下の情報をもとに質問に答えてください。
    あなたは厳密なリサーチアシスタントです。
    以下の「参照メモ」に書かれている内容【だけ】を根拠に、日本語で簡潔に答えてください。
    参照に無い内容は推測せず、「手元のメモには情報がありません」と述べてください。

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
print(answer)