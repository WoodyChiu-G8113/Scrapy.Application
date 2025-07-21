# RAG 應用主程式
# 用於寫入 PDF 資料至向量資料庫 Chroma
# 載入環境變數
from dotenv import load_dotenv
import os

load_dotenv()  # 這行會自動載入 .env 檔案
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 0: 檢查 API Key 是否存在
if not OPENAI_API_KEY:
    print("[錯誤] 未設定 OPENAI_API_KEY，無法初始化 Embeddings。請設定環境變數或在程式中提供 API Key。")
    exit(1)

# Step 1: 讀取 PDF 並分頁
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "114news_Q1.pdf"

# 載入 PDF 並分割頁面
loader = PyPDFLoader(PDF_PATH)
pages = loader.load_and_split()

# Step 2: 斷詞（ckip-transformers）
from ckip_transformers.nlp import CkipWordSegmenter

# 初始化斷詞器（預設使用 CPU）
ws_driver = CkipWordSegmenter()

# 取第一頁文本
first_page_text = pages[0].page_content if hasattr(pages[0], 'page_content') else str(pages[0])

# 進行斷詞
ws_result = ws_driver([first_page_text])

# Step 3: 向量化並寫入 Chroma
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
import os
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
index_creator = VectorstoreIndexCreator(
            embedding=embeddings,
            vectorstore_cls=Chroma,
            vectorstore_kwargs={"persist_directory":"./vector"}
)
docsearch = index_creator.from_loaders([loader])
docsearch.vectorstore.persist()
docsearch