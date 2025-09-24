数据集地址：https://modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR/summary?spm=a2c6h.12873639.article-detail.19.49c921405SSA3m



step1:下载数据集

step2:处理数据集为聊天模板的格式

step3:微调qwen2.5多模态模型

step4:预测模型



模型是采用了lora进行微调。具体参数如下：

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)
```

模型大小为 3.9B；





huggingface地址：https://huggingface.co/zhupipi/qwen8b_Medical_Finetune

参考学习：[https://developer.aliyun.com/article/1643805](https://developer.aliyun.com/article/1643805)