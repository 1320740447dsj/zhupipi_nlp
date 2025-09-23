import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from step_1_pretrain import VLMConfig, VLM
from dataset import  SFTDataset
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}


if __name__ == '__main__':
    config = VLMConfig()
    processor = AutoProcessor.from_pretrained("./siglip-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained('./Qwen2.5-0.5B-Instruct')
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    model = AutoModelForCausalLM.from_pretrained('./dsj_mutimodel/pretrain/checkpoint-3000')
    
    for name, param in model.named_parameters():
        if 'linear' in name or 'vision_model':
            param.requires_grad = False
        if 'llm_model' in name:
            param.requires_grad = True
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}') 
    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}') 
    images_path = './sft_images/sft_images'
    data_path = './sft_data.jsonl'
    output_dir = './dsj_mutimodel/sft'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SFTDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('dsj_mutimodel/sft')
    trainer.save_state()
