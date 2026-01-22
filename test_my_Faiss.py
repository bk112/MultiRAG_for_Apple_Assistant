# test_faiss_retrieval.py

import os
import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------------
# 1. 配置路径和参数
# -------------------------------
MODEL_NAME = "E:/model/bert-base-chinese"
DIMENSION = 512

# 索引路径（与你创建时一致）
INDEX_DIR = "./multimodal_rag_system_output/data_storage/vector_indices"

TEXT_INDEX_PATH = os.path.join(INDEX_DIR, "my_phone_IP_text_vector_index.faiss")
IMAGE_INDEX_PATH = os.path.join(INDEX_DIR, "my_phone_IP_image_vector_index.faiss")
MEAN_INDEX_PATH = os.path.join(INDEX_DIR, "my_phone_IP_mean_vector_index.faiss")

# 可选：Top-K 返回结果数
K = 3

# -------------------------------
# 2. 加载 CLIP 模型和处理器
# -------------------------------
print("Loading CLIP model for encoding queries...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------------------
# 3. 加载 Faiss 索引
# -------------------------------
def load_index(path):
    if not os.path.exists(path):
        print(f"❌ 索引文件不存在: {path}")
        return None
    index = faiss.read_index(path)
    print(f"✅ 已加载索引: {path} (向量数: {index.ntotal}, 维度: {index.d})")
    return index

text_index = load_index(TEXT_INDEX_PATH)
image_index = load_index(IMAGE_INDEX_PATH)
mean_index = load_index(MEAN_INDEX_PATH)

if not text_index or not image_index or not mean_index:
    raise FileNotFoundError("请确保三个 .faiss 文件都存在。")

# -------------------------------
# 4. 编码函数
# -------------------------------
def encode_text(text: str):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features.cpu().numpy().astype('float32')

def encode_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片未找到: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.cpu().numpy().astype('float32')

def encode_multimodal(text: str, image_path: str):
    text_vec = encode_text(text)
    image_vec = encode_image(image_path)
    mean_vec = (text_vec + image_vec) / 2
    return mean_vec

# -------------------------------
# 5. 检索函数
# -------------------------------
def search_text_query(query_text: str, k=K):
    """输入文本，检索最匹配的文本条目"""
    print(f"\n🔍 文本查询: '{query_text}'")
    query_vec = encode_text(query_text)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    distances, indices = text_index.search(query_vec, k)

    print("Top 匹配结果:")
    for i, (idx, ip) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue  # 无效结果

        print(f"  {i+1}. 向量ID={idx}, IP值={ip:.3f}")
    return indices[0], distances[0]

def search_image_query(image_path: str, k=K):
    """输入图像，检索最匹配的图像条目"""
    print(f"\n🔍 图像查询: '{image_path}'")
    query_vec = encode_image(image_path)
    distances, indices = image_index.search(query_vec, k)

    print("Top 匹配结果:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        similarity = 1 / (1 + dist)
        print(f"  {i+1}. 向量ID={idx}, L2距离={dist:.3f}, 相似度≈{similarity:.3f}")
    return indices[0], distances[0]

def search_multimodal_query(text: str, image_path: str, k=K):
    """输入图文组合，检索最匹配的融合条目"""
    print(f"\n🔍 多模态查询: 文本='{text}', 图像='{image_path}'")
    query_vec = encode_multimodal(text, image_path)
    distances, indices = mean_index.search(query_vec, k)

    print("Top 匹配结果:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        similarity = 1 / (1 + dist)
        print(f"  {i+1}. 向量ID={idx}, L2距离={dist:.3f}, 相似度≈{similarity:.3f}")
    return indices[0], distances[0]

# -------------------------------
# 6. 示例测试
# -------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("✅ Faiss 检索测试开始")
    print("="*60)

    # 示例 1: 纯文本查询
    search_text_query("给我推荐一个手机")

    # # 示例 2: 纯图像查询（替换为你的图片）
    # if os.path.exists("./images/beer1.jpg"):
    #     search_image_query("./images/beer1.jpg")
    # #
    # # # 示例 3: 多模态查询
    # if os.path.exists("circuit.png"):
    #     search_multimodal_query(
    #         text="an electronic circuit with resistors and capacitors",
    #         image_path="circuit.png"
    #     )
    #
    # print("\n✅ 测试完成！")