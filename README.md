环境配置：
pip install datasets

pip install accelerate

pip install deepspeed

pip install bitsandbytes

pip install jsonlines

pip install sentence_transformers

pip install dashscope

pip install -U "transformers>=4.40.0" "peft>=0.10.0" "bitsandbytes>=0.43.0" accelerate

pip install gradio 

# 每部分功能 #

## 生成数据： ##

self_instruct_generator.py：迭代生成数据，会将生成的数据存储到\dataset\raw_generator.jsonl

self_instruct_fliter.py：将raw_generator.jsonl的数据去重后存储到\dataset\final_dataset.jsonl

\dataset\filtered_seeds.jsonl: 存储用于迭代的初始种子数据。

\dataset\final_dataset.jsonl: 最终用于训练的指令数据

text2vec-large-chinese文件: 用于语义去重的模型，请从huggingface上拉下来 ： https://huggingface.co/GanymedeNil/text2vec-large-chinese

## 训练： ##
 
Qwen1.5-1.8B文本：用于训练的原版小模型，请从huggingface上拉下来  https://huggingface.co/Qwen/Qwen1.5-1.8B

train_lora_sft.py：用于进行QLora指令微调的训练代码

## 交互： ##

gradio_chat.py : 基于gradio的交互
