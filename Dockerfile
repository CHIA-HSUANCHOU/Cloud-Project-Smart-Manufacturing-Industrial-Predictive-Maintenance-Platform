# 選擇 Python 3.12 slim 作為基礎環境
FROM python:3.12-slim
# 設定工作目錄 /app
WORKDIR /app
# 安裝系統編譯工具
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# 複製 requirements.txt
COPY requirements.txt .
# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt
# 複製 Streamlit 程式和模型檔案
COPY . .
# 宣告 8080 port
EXPOSE 8080
# 啟動 Streamlit app
CMD ["sh", "-c", "streamlit run firstpage.py --server.port=8080 --server.address=0.0.0.0"]
