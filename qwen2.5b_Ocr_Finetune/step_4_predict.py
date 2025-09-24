import torch
from peft import PeftModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
from qwen_vl_utils import process_vision_info
from PIL import Image

def predict(messages, model, processor, tokenizer, max_length=8192):
    """
    使用模型进行预测
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_length)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# 模型的本地路径和微调后的模型权重路径
local_model_path = "./Qwen2.5-VL-3B-Instruct"
peft_model_path = "./Qwen2.5-VL-3B-LatexOCR/checkpoint-189"
prompt = "你是一个LaText OCR助手,目标是读取用户输入的照片，转换成LaTex公式。"


# 加载基础模型和微调权重
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 加载分词器和处理器
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path, use_fast=False, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(local_model_path)

# # 将LoRA权重加载到基础模型上
model = PeftModel.from_pretrained(model, peft_model_path)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量：{total_params / 1e6:.2f} M")
# 示例图片路径
image_path = "img_1.png"
# 确保example_image.png文件存在且可访问

# 创建输入消息
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# 进行预测
with torch.no_grad():
    prediction = predict(
        messages=messages,
        model=model,
        processor=processor,
        tokenizer=tokenizer
    )

print(f"预测的 LaTeX 公式为: {prediction}")