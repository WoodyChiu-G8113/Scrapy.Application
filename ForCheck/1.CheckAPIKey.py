
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 0: 檢查 API Key 是否存在
if not OPENAI_API_KEY:
    print("[錯誤] 未設定 OPENAI_API_KEY，無法初始化 Embeddings。請設定環境變數或在程式中提供 API Key。")
    exit(1)

# 使用OpenAI 基本 API 確認功能是否正常
def check_openai_api():
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        models = client.models.list()   
        if models.data:
            for model in models.data:
                print(f"模型名稱: {model.id}")
        else:
            print("[錯誤] OpenAI API Key 驗證失敗，未能列出任何模型。")
    except Exception as e:
        print(f"[錯誤] OpenAI API 驗證失敗：{e}")
        exit(1)


check_openai_api()
