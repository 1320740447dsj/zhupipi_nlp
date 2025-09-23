数据集来自训练minimind_dataset，通过网盘分享的文件：finetune_medical.zip
链接: https://pan.baidu.com/s/10SLrVAOhbe4EDOcUSs6dFw 提取码: fq84 
--来自百度网盘超级会员v6的分享

数据集参考链接：https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/tree/main



step1:进行微调训练

step2:进行预测



模型是采用了lora进行微调。具体参数如下：

```python
lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
```

模型大小为8.2B；

huggingface地址：https://huggingface.co/zhupipi/qwen8b_Medical_Finetune



参考学习：https://github.com/wyf3/llm_related