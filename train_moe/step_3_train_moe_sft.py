from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from dataset import SFTDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model import  LLM
from model_config import Config

if __name__ == '__main__':
    AutoConfig.register("moe_model", Config)
    AutoModelForCausalLM.register(Config, LLM)
    model = AutoModelForCausalLM.from_pretrained('./dsj_model/moe')
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./dsj_model/tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./dsj_model/moe-sft',
                            num_train_epochs=5,
                            do_train=True,
                            per_device_train_batch_size=16,
                            gradient_accumulation_steps=2,
                            # max_steps=15000,
                            logging_steps=300,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False)
    dataset = SFTDataset('./minimind_dataset/sft_1024.jsonl', tokenizer=tokenizer, max_seq_len=1024)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./dsj_model/moe-sft')
    trainer.save_state()
