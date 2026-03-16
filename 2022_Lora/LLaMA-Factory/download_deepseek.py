import os
from huggingface_hub import snapshot_download

# 设置镜像源（双重保险）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    local_dir="/DataSet/Hugging-face/DeepSeek-R1-1.5B",
    local_dir_use_symlinks=False,
    resume_download=True,
    token=None  # 公开模型不需要 token
)
print("下载完成！模型保存在: /DataSet/Hugging-face/DeepSeek-R1-1.5B")
