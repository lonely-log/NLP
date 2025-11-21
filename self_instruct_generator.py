# -*- coding: utf-8 -*-
"""
Self-Instruct 核心生成脚本
种子指令扩展 + 回答生成 → raw_generated.jsonl
"""

# -*- coding: utf-8 -*-
import json
import random
from dashscope import Generation
import dashscope
import os
import time
# ========== 语义去重模型加载 ==========
from sentence_transformers import SentenceTransformer, util

# 本地模型路径
EMBED_MODEL_PATH = "./text2vec-large-chinese"

# 相似度阈值
SIMILARITY_THRESHOLD = 0.85

# 加载句向量模型（全局加载一次）

print("加载语义去重模型 text2vec-large-chinese ...")
embedding_model = SentenceTransformer(EMBED_MODEL_PATH)
print("模型加载完成")


# ================== 配置 ==================
dashscope.api_key = "sk-7ba2fb93588a45a3b5b120658f45dc79"

# 已过滤的高质量数据，用作种子
SEED_FILE = "dataset/filtered_seeds.jsonl"

# 输出文件
OUTPUT_FILE = "dataset/raw_generated.jsonl"

# 每轮生成数量
NUM_TO_GENERATE = 6

# 抽样种子数量
NUM_SEED_SAMPLE = 5

# 每条生成请求间隔（秒）
REQUEST_INTERVAL = 1.0

# 最大迭代轮数
MAX_ITERATIONS = 3

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
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def deduplicate(data_list):
    """
    使用句向量 + 余弦相似度进行语义去重
    """
    if not data_list:
        return []

    # 提取所有 instruction 文本
    instructions = [d.get("instruction", "") for d in data_list]

    # 计算所有 instruction 向量
    embeddings = embedding_model.encode(instructions, convert_to_tensor=True)

    unique_indices = []
    for i, emb in enumerate(embeddings):
        is_dup = False
        for j in unique_indices:
            sim = util.cos_sim(emb, embeddings[j]).item()
            if sim >= SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            unique_indices.append(i)

    # 返回去重后的数据
    return [data_list[i] for i in unique_indices]


# ================== 生成函数 ==================
def generate_new_tasks(seed_tasks, num_to_generate):
    generated = []
    for _ in range(num_to_generate):
        seed = random.choice(seed_tasks)
        prompt = f"""
        你是一个严格的 JSON 输出机器。
        你只能输出 JSON，不允许任何解释、前缀、后缀。

        请按照以下格式回答（必须严格遵守）：
        {{
          "instruction": "...",
          "input": "...",
          "output": "..."
        }}

        根据这个任务，生成一个新的法律指令数据：
        指令: {seed['instruction']}
        输入: {seed.get('input', '')}

        只输出 JSON，不要输出任何多余文本。
        """

        try:
            resp = Generation.call(
                model="qwen2.5-72b-instruct",
                prompt=prompt
            )
            text = resp["output"]["text"].strip()
            # 尝试解析为 JSON
            new_task = json.loads(text)
            if "instruction" in new_task and "output" in new_task:
                # 确保 input 字段存在
                if "input" not in new_task:
                    new_task["input"] = ""
                generated.append(new_task)
        except Exception as e:
            print(f"生成失败: {e}")
        time.sleep(REQUEST_INTERVAL)
    return generated

# ================== 主流程 ==================
if __name__ == "__main__":
    # 读取已有种子
    seed_data = load_jsonl(SEED_FILE)
    if not seed_data:
        raise ValueError("种子文件为空，请先准备高质量过滤数据")

    for iteration in range(MAX_ITERATIONS):
        print(f"=== 第 {iteration+1} 轮生成 ===")
        # 抽样若干种子
        sampled_seeds = random.sample(seed_data, min(NUM_SEED_SAMPLE, len(seed_data)))
        # 生成新任务
        new_tasks = generate_new_tasks(sampled_seeds, NUM_TO_GENERATE)
        print(f"生成 {len(new_tasks)} 条新任务")
        # 去重
        new_tasks = deduplicate(new_tasks)
        # 保存到文件
        save_jsonl(OUTPUT_FILE, new_tasks)
        print(f"已保存到 {OUTPUT_FILE}")
        # 下一轮可以使用新生成的任务作为种子
        seed_data.extend(new_tasks)
        seed_data = deduplicate(seed_data)
