from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()  # 這行會自動載入 .env 檔案
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

db = Chroma(persist_directory='./vector', embedding_function=embeddings)

# 取得所有 document 資料
result = db.get()
documents = result['documents']

# 印出前幾筆
for i, doc in enumerate(documents[:5]):
    print(f"第 {i+1} 筆：", doc)
