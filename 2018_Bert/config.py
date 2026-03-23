import os

# HuggingFace 统一缓存目录（所有模型/数据集存到 hub/ 下）
HF_HOME = "/home/lnick/DataSet/Hugging-Face"
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 大文件统一目录（训练输出等）
BIG_FILES_DIR = os.path.join(HF_HOME, "2018_Bert")
OUTPUT_BASE = os.path.join(BIG_FILES_DIR, "output")
LOG_BASE = os.path.join(BIG_FILES_DIR, "logs")
os.makedirs(BIG_FILES_DIR, exist_ok=True)

# 模型配置
MODEL_NAME = "bert-base-uncased"
