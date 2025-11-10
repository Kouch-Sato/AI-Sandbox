from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm_model = "gpt-4o"
embedding_model = "text-embedding-3-large"

llm = ChatOpenAI(
  api_key = os.getenv("OPENAI_API_KEY"),
  model_name = "gpt-4o",
  temperature = 0,    
)

emb = OpenAIEmbeddings(
  api_key = os.getenv("OPENAI_API_KEY"),
  model = embedding_model,
)

# VetrorStore: Chromaのようにデータを保存する場所
from langchain_chroma import Chroma
from langchain_core.documents import Document

texts = [
  "私はオランダで生まれて、東京で育ちました",
  "趣味はボードゲームとポーカーです。",
  "AIやスタートアップの世界にワクワクするタイプです。",
  "休日はカフェで読書することが多い。",
  "猫を飼っていて、毎朝一緒に起きます。",
  "なみしろさんという方と結婚しています"
]

docs = [
  Document(page_content=t, metadata={"id": f"doc_{i}"})
   for i, t in enumerate(texts)
   ]


vectorstore = Chroma(
  collection_name = "test_collection", 
  embedding_function = emb,
  persist_directory = "./chroma_db",
)

if vectorstore._collection.count() == 0:
  vectorstore.add_documents(docs)
  vectorstore.persist()


# Retriever: 検索のためのもの
retriever = vectorstore.as_retriever(search_kwargs = { "k": 3 })


# Promopt: LLMに与える指示
USER_TEMPLATE = """
  以下の情報をもとに質問に答えてください。
  あなたは厳密なリサーチアシスタントです。
  以下の「参照メモ」に書かれている内容【だけ】を根拠に、日本語で簡潔に答えてください。
  参照に無い内容は推測せず、「手元のメモには情報がありません」と述べてください。
  回答の最後に、使った根拠の番号を [1] のように列挙してください。

  参考メモ: {context}
  質問: {question}
"""

prompt = ChatPromptTemplate.from_messages([
  ("system", "あなたは厳密なリサーチアシスタントです。"),
  ("user", USER_TEMPLATE)
])

def format_docs_with_numbers(docs):
  contexts = []
  for i, doc in enumerate(docs):
    contexts.append(f"[{i+1}] {doc.page_content}")
  return "\n".join(contexts)


# RAG Chainの実装
from langchain_core.runnables import RunnableLambda

rag_chain = (
  {
    "context": RunnableLambda(lambda x: format_docs_with_numbers(retriever.invoke(x["question"]))),
    "question": RunnablePassthrough(),
  }
  | prompt
  | llm
  | StrOutputParser()
)

def rag_answer(question: str) -> str:
  return rag_chain.invoke({ "question": question })

print("質問を入力してください")
user_question = input("質問: ")
answer = rag_answer(user_question)
print("回答:")
print(answer)