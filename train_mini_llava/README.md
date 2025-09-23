数据地址：通过网盘分享的文件：mutimode_dataset.zip
链接: https://pan.baidu.com/s/1-5Og2tsbUSrbKNUuluGuLg 提取码: 3v8v 
--来自百度网盘超级会员v6的分享



预训练模型较为“粗糙”，一般是对图片的描述，不过我在查看数据集发现，很多数据质量并不高

微调训练主要是对图片中的细节进行描述，比如图中有几个人，这些人在干嘛之类的。



step1 进行预训练

step2 进行sft训练

gradio_visulize进行可视化结果，测试结果表面经过sft的结果比只经过预训练的结果要好，因此我只上传了sft后的模型权重

模型是qwen2.5-0.5B模型和siglip-base-patch16-224，两者融合方式参考的llava模型。

模型大小：0.7B



huggingface地址：git clone https://huggingface.co/zhupipi/mini_llava

参考：https://github.com/wyf3/llm_related
