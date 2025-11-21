# -*- coding: utf-8 -*-
import json
from sentence_transformers import SentenceTransformer, util
import os

# ================== 内置参数 ==================
INPUT_FILE = "dataset/raw_generated.jsonl"    # 待过滤的原始数据
OUTPUT_FILE = "dataset/final_dataset.jsonl"       # 输出文件
SIMILARITY_THRESHOLD = 0.85                               # 语义相似阈值

# ================== 工具函数 ==================
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    return data

def save_jsonl(file_path, data_list):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def basic_filter(task):
    """基础过滤：非空 instruction/output"""
    if "instruction" not in task or not task["instruction"].strip():
        return False
    if "output" not in task or not task["output"].strip():
        return False
    return True

# ================== 加载数据 ==================
data_list = load_jsonl(INPUT_FILE)
print(f"原始数据量: {len(data_list)} 条")

# 基础过滤
data_list = [d for d in data_list if basic_filter(d)]
print(f"基础过滤后数据量: {len(data_list)} 条")

# ================== 语义去重 ==================
print("开始语义去重...")
model = SentenceTransformer("./text2vec-large-chinese")
instructions = [d["instruction"] for d in data_list]
embeddings = model.encode(instructions, convert_to_tensor=True)

unique_indices = []
for i, emb in enumerate(embeddings):
    is_duplicate = False
    for j in unique_indices:
        sim = util.cos_sim(emb, embeddings[j]).item()
        if sim >= SIMILARITY_THRESHOLD:
            is_duplicate = True
            break
    if not is_duplicate:
        unique_indices.append(i)

filtered_data = [data_list[i] for i in unique_indices]
print(f"语义去重后数据量: {len(filtered_data)} 条")

# ================== 保存结果 ==================
save_jsonl(OUTPUT_FILE, filtered_data)
print(f"过滤完成，已保存到 {OUTPUT_FILE}")
