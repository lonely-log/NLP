import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "./Qwen1.5-1.8B"
LORA_PATH = "model_result/qwen_lora_sft_3h/checkpoint-8000"


# ===========================
# 加载模型 (Base + LoRA)
# ===========================
def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Applying LoRA weights...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        device_map="auto"
    )

    model.eval()
    print("Model loaded!")
    return tokenizer, model


tokenizer, model = load_model()


# ===========================
# 推理函数（多轮对话）
# ===========================
def predict(history, user_input):
    """
    Gradio history: List[List[str, str]]
    Our format:     [(user, assistant), ...]
    """
    # 构建 prompt（拼接历史）
    prompt = ""
    for user, assistant in history:
        prompt += f"用户：{user}\n助手：{assistant}\n"

    prompt += f"用户：{user_input}\n助手："

    # 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05
        )

    # 解码结果
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 截取 assistant 部分
    answer = full_output[len(prompt):].strip()

    # 更新 history
    history.append((user_input, answer))
    return history, history


# ===========================
# Gradio UI
# ===========================
def clear_history():
    return [], []


with gr.Blocks(title="LoRA-Qwen Chat UI") as demo:
    gr.Markdown("<h2><center>🧠 LoRA 微调 Qwen 对话界面</center></h2>")

    chatbot = gr.Chatbot(height=500)
    user_input = gr.Textbox(
        label="你的输入",
        placeholder="请输入你的问题…",
    )

    submit_btn = gr.Button("发送")
    clear_btn = gr.Button("清空对话")

    submit_btn.click(
        predict,
        inputs=[chatbot, user_input],
        outputs=[chatbot, chatbot]
    )

    clear_btn.click(clear_history, outputs=[chatbot, chatbot])


# ===========================
# 启动服务
# ===========================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
