from transformers import  AutoTokenizer
from transformers import AutoProcessor
import torch
from transformers import Trainer, TrainingArguments
from typing import List, Dict, Any
from dataset import  MyDataset
from model_config import  VLMConfig
from model import VLM
     

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
    config = VLMConfig(vision_model_path='./siglip-base-patch16-224', image_pad_num=49)
    model = VLM(config).cuda()
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = './LLaVA-CC3M-Pretrain-595K/images'
    data_path = './Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'dsj_mutimodel/pretrain' 
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=16,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('dsj_mutimodel/pretrain')
    trainer.save_state()
    
    

    
    