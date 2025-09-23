import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
# -------------------------------------------------------------------------

# 加载微调后的模型
base_model_path = "./Qwen3-8B"
lora_adapter_path = "./checkpoint-2270"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 加载LoRA适配器到模型
model = PeftModel.from_pretrained(model, lora_adapter_path)

# 定义对话输入
messages = [
    {"role": "system", "content": PROMPT},
    {"role": "user", "content": "我想了解一下高血压的常见症状。"}
]

# 调用 predict 函数生成结果
output_text = predict(messages, model, tokenizer)
print("生成的回答：")
print(output_text)

# 你可以继续和模型进行多轮对话
# messages.append({"role": "assistant", "content": output_text})
# messages.append({"role": "user", "content": "那日常生活中我应该注意些什么？"})
# next_output = predict(messages, model, tokenizer)
# print("多轮对话生成的回答：")
# print(next_output)