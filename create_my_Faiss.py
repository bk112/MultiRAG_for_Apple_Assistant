# create_my_faiss_rag.py
import json
import os
import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------------
# 1. 配置参数
# -------------------------------
MODEL_NAME = "./clip_model_cache"  # CLIP 模型，输出 512 维向量
DIMENSION = 512  # 向量维度

# Faiss 索引保存路径
INDEX_DIR = "./multimodal_rag_system_output/data_storage/vector_indices"
os.makedirs(INDEX_DIR, exist_ok=True)

TEXT_INDEX_PATH = os.path.join(INDEX_DIR, "my_phone_IP_text_vector_index.faiss")
IMAGE_INDEX_PATH = os.path.join(INDEX_DIR, "my_phone_IP_image_vector_index.faiss")
MEAN_INDEX_PATH = os.path.join(INDEX_DIR, "my_phone_IP_mean_vector_index.faiss")

# 示例数据：文本-图片对列表
# 请确保图片路径存在！可以替换成你自己的数据
RAG_DATA_FILE = "./RAG_data/mobilePhone.json"  # 可改为你的实际路径
# 从文件读取图文对
data_pairs = []
with open(RAG_DATA_FILE, 'r', encoding='utf-8') as f:
    data_pairs = json.load(f)

# -------------------------------
# 2. 加载 CLIP 模型和处理器
# -------------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# 使用 CPU，如有 GPU 可启用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------------------
# 3. 创建 Faiss 索引（Flat L2 相似度）
# -------------------------------
print("Creating Faiss indexes...")

def load_or_create_index(path, dimension):
    if os.path.exists(path):
        print(f"📂 加载已有索引: {path}")
        index = faiss.read_index(path)
        print(f"✅ 当前向量数: {index.ntotal}, 维度: {index.d}")
    else:
        print(f"🆕 创建新索引: {path}")
        index = faiss.IndexFlatIP(dimension)  # 使用 IP 距离
    return index

text_index = load_or_create_index(TEXT_INDEX_PATH, DIMENSION)
image_index = load_or_create_index(IMAGE_INDEX_PATH, DIMENSION)
mean_index = load_or_create_index(MEAN_INDEX_PATH, DIMENSION)

print(f"Text index (ntotal): {text_index.ntotal}")
print(f"Image index (ntotal): {image_index.ntotal}")
print(f"Mean index (ntotal): {mean_index.ntotal}")

# -------------------------------
# 4. 处理每一对文本-图片，生成向量并添加到索引
# -------------------------------
ids = []  # 存储 ID（可扩展为元数据）
for i, pair in enumerate(data_pairs):
    text = pair["text"]
    img_path = pair["image_path"]

    print(f"\nProcessing pair {i+1}: {text}")

    # --- 文本向量化 ---
    inputs_text = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs_text)
    text_vec = text_features.cpu().numpy().astype('float32')
    text_vec = text_vec / np.linalg.norm(text_vec, axis=1, keepdims=True)

    # --- 图像向量化 ---
    if not os.path.exists(img_path):
        print(f"⚠️  Image not found: {img_path}, skipping...")
        continue

    image = Image.open(img_path).convert("RGB")
    inputs_image = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs_image)
    image_vec = image_features.cpu().numpy().astype('float32')
    image_vec = image_vec / np.linalg.norm(image_vec, axis=1, keepdims=True)

    # --- 归一化（CLIP 向量通常已归一化，但 Faiss 中常使用内积搜索，这里保持 L2）---
    # 注意：CLIP 输出的是归一化向量，所以 L2 距离 ≈ 2 - 2*cosine，适合语义相似度

    # --- 添加到索引 ---
    text_index.add(text_vec)
    image_index.add(image_vec)

    # --- 计算平均向量 ---
    mean_vec = (text_vec + image_vec) / 2
    mean_vec = mean_vec / np.linalg.norm(mean_vec, axis=1, keepdims=True)
    mean_index.add(mean_vec)

    ids.append(i)
    print(f"✅ Added vector {i} to all indexes.")

# -------------------------------
# 5. 保存索引到文件
# -------------------------------
print("\nSaving indexes to disk...")
faiss.write_index(text_index, TEXT_INDEX_PATH)
faiss.write_index(image_index, IMAGE_INDEX_PATH)
faiss.write_index(mean_index, MEAN_INDEX_PATH)

print(f"✅ Text index saved to: {TEXT_INDEX_PATH}")
print(f"✅ Image index saved to: {IMAGE_INDEX_PATH}")
print(f"✅ Mean index saved to: {MEAN_INDEX_PATH}")

# -------------------------------
# 6. 验证：读取并检查数量
# -------------------------------
print("\nValidating saved indexes:")
loaded_text_index = faiss.read_index(TEXT_INDEX_PATH)
loaded_image_index = faiss.read_index(IMAGE_INDEX_PATH)
loaded_mean_index = faiss.read_index(MEAN_INDEX_PATH)

print(f"Loaded text index - Total vectors: {loaded_text_index.ntotal}")
print(f"Loaded image index - Total vectors: {loaded_image_index.ntotal}")
print(f"Loaded mean index - Total vectors: {loaded_mean_index.ntotal}")

print("\n✅ All done! You now have a local multi-modal RAG with 3 Faiss indexes.")