import os
from huggingface_hub import hf_hub_download, login
from dotenv import load_dotenv
from pyspark.sql import SparkSession

def download_model():
    # Khởi tạo Spark Session
    spark = SparkSession.builder.appName("ModelDownloader").getOrCreate()
    
    # Load biến môi trường từ .env (nếu có)
    load_dotenv()
    # Lấy HF_TOKEN từ biến môi trường
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        raise ValueError("Vui lòng đặt biến môi trường HF_TOKEN!")

    # Đăng nhập vào Hugging Face
    login(token=HF_TOKEN)
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        local_dir="D:\TestChatBotAPIWITHSpark"
    )
    
    print(f"Mô hình đã tải về: {model_path}")
    
    # Dừng Spark Session
    spark.stop()

if __name__ == "__main__":
    download_model()