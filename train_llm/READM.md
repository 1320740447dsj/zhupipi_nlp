数据集来自训练minimind_dataset，百度网盘地址为：通过网盘分享的文件：minimind_dataset.zip
链接: https://pan.baidu.com/s/1Csh1kIW_HqH_7SJ1su5x6A 提取码: bai8 
--来自百度网盘超级会员v6的分享



self_cognition.jsonl是模型自我认知数据集：
不过我目前还不知道在哪个训练阶段添加自我认知数据集，我尝试在step4进行，不过会导致大模型遗忘；在step3进行，我发现self_cognition.jsonl的数据量不太够。

step1:进行tokenizer训练

step2:进行pretrain预训练

step3:进行监督微调

step4:进行强化学习



test_llm.ipynb进行测试文件



模型是采用了旋转位置编码、分组查询注意力、FlashAttention等。具体参数如下：

```python
hidden_size=512,
 num_attention_heads=16,
 num_key_value_heads=8,
 flash_attn=True,
 attention_bias=False,
 max_seq_len=512,
 intermediate_size=2048,
 mlp_bias=False,
 vocab_size=6400,
 n_layers=8,
 dropout=0.0,
```

模型大小为0.03B；

模型权重将上传到huggingface地址：https://huggingface.co/zhupipi/litelm