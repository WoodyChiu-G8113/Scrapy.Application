# 載入環境變數
from dotenv import load_dotenv
import os

load_dotenv()  # 這行會自動載入 .env 檔案
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 0: 檢查 API Key 是否存在
if not OPENAI_API_KEY:
    print("[錯誤] 未設定 OPENAI_API_KEY，無法初始化 Embeddings。請設定環境變數或在程式中提供 API Key。")
    exit(1)

## Step04: 連接資料庫：Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db = Chroma(persist_directory='./vector', embedding_function=embeddings)
db

## Step05: 對話查詢：RetrievalQA -> create_retrieval_chain (RAG)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

retriever = db.as_retriever(search_kwargs={"k":2})

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "請根據上下文來回答問題，不知道答案就回答不知道不要試圖編造答案"
    "你是一個專業的股票經紀理人，且不具備其他領域知識"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)
# retriever = 資料從哪裡來？ = {context}
# question_answer_chain = 你要怎麼查資料？
chain

# Step06: 問答測試
print(chain.invoke({"input": '你知道嘉義在哪裡？'})['answer'])
print(chain.invoke({"input": '你的資料有甚麼資訊?你知道旅遊相關資訊嗎?'})['answer'])