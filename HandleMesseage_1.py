import requests
from underthesea import word_tokenize
import re
import os
import pandas as pd
from llama_cpp import Llama
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Khởi tạo Spark Session
spark = SparkSession.builder.appName("ChatBotAPI_Spark").getOrCreate()

# PATH LOCAL
MODEL_PATH = "D:\TestChatBotAPIWITHSpark\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
EXCEL_PATH = "D:\TestChatBotAPIWITHSpark\ListAPIChatBot.xlsx"

# Kiểm tra file mô hình và Excel
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình tại {MODEL_PATH}!")
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy file Excel tại {EXCEL_PATH}!")
print(f"Đang sử dụng mô hình từ: {MODEL_PATH}")
print(f"Đang đọc dữ liệu API từ: {EXCEL_PATH}")

# Khởi tạo Llama
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=10240,
    n_threads=10,
    n_gpu_layers=40,
    verbose=False
)

# Đọc file Excel bằng Spark
api_df = spark.read.format("excel").option("header", "true").load(EXCEL_PATH)

# Tokenize
def preprocess_message(message):
    message = message.lower()
    tokens = word_tokenize(message)
    return tokens

# Phân luồng API
def detect_intent(tokens):
    if "thống kê" in tokens and ("qr" in tokens or "gán mã" in tokens):
        return "/service/gettotalsite"
    return None

# Hàm xử lý Spark DataFrame
def filter_api_info(intent):
    filtered_df = api_df.filter(col("Name") == intent)
    return filtered_df.collect()[0] if filtered_df.count() > 0 else None

# Hàm tạo câu trả lời từ Llama
def get_message(messages):
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0,
        top_p=1,
        stop=["<|eot_id|>"]
    )
    content = response["choices"][0]["message"]["content"].strip()
    return re.sub(r'<think>.*?</think>|[\u4e00-\u9fff]|<think>', '', content, flags=re.DOTALL).strip()

# Hàm tạo câu trả lời với Apache Spark
def generate_response(messages_client, messages_server, intent=None):
    if not intent:
        tokens = preprocess_message(messages_client)
        intent = detect_intent(tokens)
    
    api_info = filter_api_info(intent)
    if api_info is None:
        return "Không tìm thấy thông tin API trong file Excel."
    
    api_result_desc = api_info["Results"]
    
    prompt_template = """
    Bạn là một trợ lý AI thông minh được thiết kế để hỗ trợ người dùng dựa trên câu hỏi và dữ liệu từ API. 
    Dưới đây là thông tin đầu vào:
    - `messages_client`: "{client_input}"
    - `messages_server`: "{server_input}"
    - Mô tả kết quả API: "{api_result_desc}"
    """
    
    prompt = prompt_template.format(
        client_input=messages_client,
        server_input=str(messages_server),
        api_result_desc=api_result_desc
    )
    
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý AI hỗ trợ dựa trên API."},
        {"role": "user", "content": prompt}
    ]
    
    response_text = get_message(messages)
    response_text = response_text.replace("\n", "").encode("utf-8").decode("utf-8")
    return response_text

if __name__ == "__main__":
    test_messages_client = "Thống kê số lượng ATS hoạt động tốt"
    messages_server = [
            {
                "a10": 5,
                "obj_def_name": "ATS",
                "obj_def_id": 10,
                "a0": "ATS",
                "a1": 6355,
                "a2": 3,
                "total": 6355,
                "a3": 0,
                "a4": 22,
                "a5": 62,
                "a6": 6214,
                "a7": 15,
                "a8": 22,
                "a9": 15
            },
            {
                "a10": 99,
                "obj_def_name": "CỘT ANTEN & CẦU CÁP OUTDOOR",
                "obj_def_id": 25,
                "a0": "CỘT ANTEN & CẦU CÁP OUTDOOR",
                "a1": 15557,
                "a2": 0,
                "total": 15557,
                "a3": 0,
                "a4": 2,
                "a5": 30,
                "a6": 15394,
                "a7": 11,
                "a8": 16,
                "a9": 5
            },
            {
                "a10": 2347,
                "obj_def_name": "PHÒNG MÁY",
                "obj_def_id": 61,
                "a0": "PHÒNG MÁY",
                "a1": 15541,
                "a2": 0,
                "total": 15541,
                "a3": 0,
                "a4": 20,
                "a5": 142,
                "a6": 12942,
                "a7": 26,
                "a8": 25,
                "a9": 39
            },
            {
                "a10": 2948,
                "obj_def_name": "PHÒNG MPĐ",
                "obj_def_id": 63,
                "a0": "PHÒNG MPĐ",
                "a1": 12389,
                "a2": 0,
                "total": 12389,
                "a3": 0,
                "a4": 21,
                "a5": 13,
                "a6": 9390,
                "a7": 3,
                "a8": 5,
                "a9": 9
            },
            {
                "a10": 112,
                "obj_def_name": "MÁY LẠNH",
                "obj_def_id": 64,
                "a0": "MÁY LẠNH",
                "a1": 26040,
                "a2": 18441,
                "total": 26040,
                "a3": 5347,
                "a4": 21,
                "a5": 1680,
                "a6": 22034,
                "a7": 1020,
                "a8": 235,
                "a9": 938
            },
            {
                "a10": 3395,
                "obj_def_name": "QUẠT THÔNG GIÓ",
                "obj_def_id": 68,
                "a0": "QUẠT THÔNG GIÓ",
                "a1": 5095,
                "a2": 1326,
                "total": 5095,
                "a3": 3182,
                "a4": 113,
                "a5": 388,
                "a6": 1054,
                "a7": 143,
                "a8": 0,
                "a9": 2
            },
            {
                "a10": 1,
                "obj_def_name": "ỔN ÁP",
                "obj_def_id": 69,
                "a0": "ỔN ÁP",
                "a1": 6181,
                "a2": 5677,
                "total": 6181,
                "a3": 412,
                "a4": 890,
                "a5": 844,
                "a6": 3642,
                "a7": 567,
                "a8": 0,
                "a9": 237
            },
            {
                "a10": 56,
                "obj_def_name": "TỦ NGUỒN",
                "obj_def_id": 70,
                "a0": "TỦ NGUỒN",
                "a1": 25680,
                "a2": 18467,
                "total": 25680,
                "a3": 4792,
                "a4": 377,
                "a5": 544,
                "a6": 24173,
                "a7": 246,
                "a8": 128,
                "a9": 156
            },
            {
                "a10": 27,
                "obj_def_name": "MÁY PHÁT ĐIỆN",
                "obj_def_id": 75,
                "a0": "MÁY PHÁT ĐIỆN",
                "a1": 10808,
                "a2": 5454,
                "total": 10808,
                "a3": 3997,
                "a4": 1,
                "a5": 403,
                "a6": 10116,
                "a7": 36,
                "a8": 117,
                "a9": 108
            },
            {
                "a10": 56,
                "obj_def_name": "THIẾT BỊ CHỐNG SÉT (SPD)",
                "obj_def_id": 78,
                "a0": "THIẾT BỊ CHỐNG SÉT (SPD)",
                "a1": 12774,
                "a2": 9186,
                "total": 12774,
                "a3": 2164,
                "a4": 19,
                "a5": 229,
                "a6": 12335,
                "a7": 81,
                "a8": 7,
                "a9": 47
            },
            {
                "a10": 3,
                "obj_def_name": "CẢNH BÁO NGOÀI",
                "obj_def_id": 80,
                "a0": "CẢNH BÁO NGOÀI",
                "a1": 8897,
                "a2": 7625,
                "total": 8897,
                "a3": 1124,
                "a4": 96,
                "a5": 3325,
                "a6": 4667,
                "a7": 592,
                "a8": 2,
                "a9": 212
            },
            {
                "a10": 19,
                "obj_def_name": "BÌNH CO2",
                "obj_def_id": 101,
                "a0": "BÌNH CO2",
                "a1": 13247,
                "a2": 10079,
                "total": 13247,
                "a3": 2268,
                "a4": 759,
                "a5": 768,
                "a6": 11076,
                "a7": 381,
                "a8": 243,
                "a9": 1
            },
            {
                "a10": 66,
                "obj_def_name": "BÌNH BỘT",
                "obj_def_id": 102,
                "a0": "BÌNH BỘT",
                "a1": 9236,
                "a2": 5278,
                "total": 9236,
                "a3": 2911,
                "a4": 782,
                "a5": 962,
                "a6": 6610,
                "a7": 692,
                "a8": 108,
                "a9": 16
            },
            {
                "a10": 54,
                "obj_def_name": "TỦ BTS 3G",
                "obj_def_id": 265,
                "a0": "TỦ BTS 3G",
                "a1": 16700,
                "a2": 10552,
                "total": 16700,
                "a3": 5357,
                "a4": 5,
                "a5": 14,
                "a6": 16608,
                "a7": 2,
                "a8": 1,
                "a9": 16
            },
            {
                "a10": 171,
                "obj_def_name": "TỦ BTS 4G",
                "obj_def_id": 266,
                "a0": "TỦ BTS 4G",
                "a1": 16981,
                "a2": 8881,
                "total": 16981,
                "a3": 6743,
                "a4": 9,
                "a5": 1,
                "a6": 16792,
                "a7": 0,
                "a8": 2,
                "a9": 6
            },
            {
                "a10": 179291,
                "obj_def_name": "CELL",
                "obj_def_id": 281,
                "a0": "CELL",
                "a1": 179291,
                "a2": 0,
                "total": 179291,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 0,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 37,
                "obj_def_name": "TỦ BTS 2G",
                "obj_def_id": 2000,
                "a0": "TỦ BTS 2G",
                "a1": 9504,
                "a2": 6537,
                "total": 9504,
                "a3": 2673,
                "a4": 653,
                "a5": 26,
                "a6": 8782,
                "a7": 6,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 46,
                "obj_def_name": "BOOSTER",
                "obj_def_id": 2003,
                "a0": "BOOSTER",
                "a1": 46,
                "a2": 0,
                "total": 46,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 0,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 245,
                "obj_def_name": "TỔ ACCU",
                "obj_def_id": 2010,
                "a0": "TỔ ACCU",
                "a1": 36494,
                "a2": 17890,
                "total": 36494,
                "a3": 9103,
                "a4": 159,
                "a5": 5187,
                "a6": 28037,
                "a7": 1777,
                "a8": 428,
                "a9": 661
            },
            {
                "a10": 0,
                "obj_def_name": "BÌNH FM200",
                "obj_def_id": 2044,
                "a0": "BÌNH FM200",
                "a1": 38,
                "a2": 25,
                "total": 38,
                "a3": 8,
                "a4": 0,
                "a5": 3,
                "a6": 35,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 1,
                "obj_def_name": "BÌNH STAT-X",
                "obj_def_id": 2045,
                "a0": "BÌNH STAT-X",
                "a1": 740,
                "a2": 162,
                "total": 740,
                "a3": 431,
                "a4": 4,
                "a5": 0,
                "a6": 735,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "TỦ BTS 5G",
                "obj_def_id": 2049,
                "a0": "TỦ BTS 5G",
                "a1": 54,
                "a2": 3,
                "total": 54,
                "a3": 44,
                "a4": 0,
                "a5": 0,
                "a6": 54,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 144,
                "obj_def_name": "TBVT",
                "obj_def_id": 2067,
                "a0": "TBVT",
                "a1": 287192,
                "a2": 7972,
                "total": 287192,
                "a3": 246011,
                "a4": 4458,
                "a5": 2301,
                "a6": 275615,
                "a7": 352,
                "a8": 1039,
                "a9": 3283
            },
            {
                "a10": 16,
                "obj_def_name": "ACCU ĐỀ",
                "obj_def_id": 2070,
                "a0": "ACCU ĐỀ",
                "a1": 5692,
                "a2": 1,
                "total": 5692,
                "a3": 0,
                "a4": 0,
                "a5": 418,
                "a6": 5049,
                "a7": 154,
                "a8": 32,
                "a9": 23
            },
            {
                "a10": 146,
                "obj_def_name": "ANTEN",
                "obj_def_id": 2084,
                "a0": "ANTEN",
                "a1": 71735,
                "a2": 1,
                "total": 71735,
                "a3": 0,
                "a4": 9,
                "a5": 2,
                "a6": 71552,
                "a7": 0,
                "a8": 26,
                "a9": 0
            },
            {
                "a10": 3126,
                "obj_def_name": "THIẾT BỊ TRUYỀN DẪN QUANG (ACCESS)",
                "obj_def_id": 2091,
                "a0": "THIẾT BỊ TRUYỀN DẪN QUANG (ACCESS)",
                "a1": 8764,
                "a2": 1629,
                "total": 8764,
                "a3": 6090,
                "a4": 101,
                "a5": 1,
                "a6": 5536,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 1383,
                "obj_def_name": "THIẾT BỊ TRUYỀN DẪN VIBA",
                "obj_def_id": 2093,
                "a0": "THIẾT BỊ TRUYỀN DẪN VIBA",
                "a1": 5128,
                "a2": 438,
                "total": 5128,
                "a3": 3567,
                "a4": 970,
                "a5": 15,
                "a6": 2759,
                "a7": 1,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 3272,
                "obj_def_name": "ANTEN VIBA",
                "obj_def_id": 2096,
                "a0": "ANTEN VIBA",
                "a1": 3272,
                "a2": 0,
                "total": 3272,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 0,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 288,
                "obj_def_name": "THIẾT BỊ TRUYỀN DẪN METRO",
                "obj_def_id": 2100,
                "a0": "THIẾT BỊ TRUYỀN DẪN METRO",
                "a1": 621,
                "a2": 0,
                "total": 621,
                "a3": 0,
                "a4": 11,
                "a5": 0,
                "a6": 322,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 41,
                "obj_def_name": "THIẾT BỊ FTTX",
                "obj_def_id": 2102,
                "a0": "THIẾT BỊ FTTX",
                "a1": 151,
                "a2": 0,
                "total": 151,
                "a3": 0,
                "a4": 1,
                "a5": 0,
                "a6": 107,
                "a7": 0,
                "a8": 2,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "PIN NLMT",
                "obj_def_id": 2103,
                "a0": "PIN NLMT",
                "a1": 1,
                "a2": 0,
                "total": 1,
                "a3": 1,
                "a4": 0,
                "a5": 0,
                "a6": 1,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "NĂNG LƯỢNG MẶT TRỜI",
                "obj_def_id": 2105,
                "a0": "NĂNG LƯỢNG MẶT TRỜI",
                "a1": 509,
                "a2": 100,
                "total": 509,
                "a3": 183,
                "a4": 0,
                "a5": 0,
                "a6": 506,
                "a7": 0,
                "a8": 3,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "BẢN QUYỀN",
                "obj_def_id": 2145,
                "a0": "BẢN QUYỀN",
                "a1": 1,
                "a2": 0,
                "total": 1,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 1,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "BIẾN ÁP",
                "obj_def_id": 2146,
                "a0": "BIẾN ÁP",
                "a1": 10,
                "a2": 0,
                "total": 10,
                "a3": 0,
                "a4": 0,
                "a5": 2,
                "a6": 6,
                "a7": 2,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "CẢM BIẾN",
                "obj_def_id": 2149,
                "a0": "CẢM BIẾN",
                "a1": 3,
                "a2": 0,
                "total": 3,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 3,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "CÔNG CỤ (THEO BỘ)",
                "obj_def_id": 2151,
                "a0": "CÔNG CỤ (THEO BỘ)",
                "a1": 97,
                "a2": 0,
                "total": 97,
                "a3": 0,
                "a4": 0,
                "a5": 22,
                "a6": 75,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 2,
                "obj_def_name": "FEEDER",
                "obj_def_id": 2153,
                "a0": "FEEDER",
                "a1": 13,
                "a2": 0,
                "total": 13,
                "a3": 0,
                "a4": 1,
                "a5": 0,
                "a6": 10,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 1,
                "obj_def_name": "HỆ THỐNG PHỤ TRỢ BTS (CSHT) (THEO BỘ)",
                "obj_def_id": 2155,
                "a0": "HỆ THỐNG PHỤ TRỢ BTS (CSHT) (THEO BỘ)",
                "a1": 2,
                "a2": 0,
                "total": 2,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 1,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "NHÀ TRẠM (THEO BỘ)",
                "obj_def_id": 2156,
                "a0": "NHÀ TRẠM (THEO BỘ)",
                "a1": 5,
                "a2": 0,
                "total": 5,
                "a3": 0,
                "a4": 1,
                "a5": 0,
                "a6": 4,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "THIẾT BỊ (THEO BỘ)",
                "obj_def_id": 2157,
                "a0": "THIẾT BỊ (THEO BỘ)",
                "a1": 64,
                "a2": 0,
                "total": 64,
                "a3": 0,
                "a4": 17,
                "a5": 1,
                "a6": 46,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "JUMPER",
                "obj_def_id": 2158,
                "a0": "JUMPER",
                "a1": 9,
                "a2": 0,
                "total": 9,
                "a3": 0,
                "a4": 3,
                "a5": 0,
                "a6": 6,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "MÁY MÓC THIẾT BỊ",
                "obj_def_id": 2160,
                "a0": "MÁY MÓC THIẾT BỊ",
                "a1": 6,
                "a2": 0,
                "total": 6,
                "a3": 0,
                "a4": 2,
                "a5": 0,
                "a6": 4,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "NGUỒN ĐIỆN",
                "obj_def_id": 2161,
                "a0": "NGUỒN ĐIỆN",
                "a1": 1,
                "a2": 0,
                "total": 1,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 1,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "VẬT TƯ",
                "obj_def_id": 2162,
                "a0": "VẬT TƯ",
                "a1": 74,
                "a2": 0,
                "total": 74,
                "a3": 0,
                "a4": 1,
                "a5": 0,
                "a6": 73,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "Ô TÔ",
                "obj_def_id": 2164,
                "a0": "Ô TÔ",
                "a1": 11,
                "a2": 0,
                "total": 11,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 11,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 1,
                "obj_def_name": "THIẾT BỊ ĐIỆN",
                "obj_def_id": 2165,
                "a0": "THIẾT BỊ ĐIỆN",
                "a1": 2,
                "a2": 0,
                "total": 2,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 1,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 1,
                "obj_def_name": "CÔNG CỤ (ĐƠN CHIẾC)",
                "obj_def_id": 2183,
                "a0": "CÔNG CỤ (ĐƠN CHIẾC)",
                "a1": 345,
                "a2": 0,
                "total": 345,
                "a3": 0,
                "a4": 8,
                "a5": 18,
                "a6": 311,
                "a7": 7,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "HỆ THỐNG PHỤ TRỢ BTS (CSHT) (ĐƠN CHIẾC)",
                "obj_def_id": 2185,
                "a0": "HỆ THỐNG PHỤ TRỢ BTS (CSHT) (ĐƠN CHIẾC)",
                "a1": 1,
                "a2": 0,
                "total": 1,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 1,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "THIẾT BỊ (ĐƠN CHIẾC)",
                "obj_def_id": 2187,
                "a0": "THIẾT BỊ (ĐƠN CHIẾC)",
                "a1": 1,
                "a2": 0,
                "total": 1,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 1,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "TRUYỀN DẪN QUANG (THEO BỘ)",
                "obj_def_id": 2188,
                "a0": "TRUYỀN DẪN QUANG (THEO BỘ)",
                "a1": 19,
                "a2": 0,
                "total": 19,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 18,
                "a7": 0,
                "a8": 1,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "TỔNG ĐÀI",
                "obj_def_id": 2205,
                "a0": "TỔNG ĐÀI",
                "a1": 165,
                "a2": 0,
                "total": 165,
                "a3": 0,
                "a4": 30,
                "a5": 0,
                "a6": 135,
                "a7": 0,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 7,
                "obj_def_name": "BỘ THIẾT BỊ CWDM",
                "obj_def_id": 2214,
                "a0": "BỘ THIẾT BỊ CWDM",
                "a1": 2842,
                "a2": 0,
                "total": 2842,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 581,
                "a7": 0,
                "a8": 2254,
                "a9": 0
            },
            {
                "a10": 65,
                "obj_def_name": "CONTROLLER",
                "obj_def_id": 2215,
                "a0": "CONTROLLER",
                "a1": 4118,
                "a2": 0,
                "total": 4118,
                "a3": 0,
                "a4": 52,
                "a5": 657,
                "a6": 3313,
                "a7": 28,
                "a8": 3,
                "a9": 0
            },
            {
                "a10": 56,
                "obj_def_name": "RECTIFIER",
                "obj_def_id": 2216,
                "a0": "RECTIFIER",
                "a1": 7292,
                "a2": 0,
                "total": 7292,
                "a3": 0,
                "a4": 48,
                "a5": 48,
                "a6": 7113,
                "a7": 21,
                "a8": 4,
                "a9": 2
            },
            {
                "a10": 8,
                "obj_def_name": "BẢNG ĐIỀU KHIỂN",
                "obj_def_id": 2217,
                "a0": "BẢNG ĐIỀU KHIỂN",
                "a1": 2084,
                "a2": 0,
                "total": 2084,
                "a3": 0,
                "a4": 4,
                "a5": 24,
                "a6": 2046,
                "a7": 2,
                "a8": 0,
                "a9": 0
            },
            {
                "a10": 0,
                "obj_def_name": "BỘ BACKUP DC LƯU ĐỘNG",
                "obj_def_id": 2219,
                "a0": "BỘ BACKUP DC LƯU ĐỘNG",
                "a1": 194,
                "a2": 10,
                "total": 194,
                "a3": 83,
                "a4": 0,
                "a5": 0,
                "a6": 189,
                "a7": 0,
                "a8": 5,
                "a9": 0
            },
            {
                "a10": 7,
                "obj_def_name": "BACKUP ACCU",
                "obj_def_id": 2222,
                "a0": "BACKUP ACCU",
                "a1": 537,
                "a2": 0,
                "total": 537,
                "a3": 0,
                "a4": 0,
                "a5": 0,
                "a6": 530,
                "a7": 0,
                "a8": 0,
                "a9": 0
            }
        ]
    test_intent = "/service/asset_dashboard"

    response = generate_response(test_messages_client, test_messages_server, test_intent)
    print("Đáp án:", response)

    # Dừng Spark
    spark.stop()






