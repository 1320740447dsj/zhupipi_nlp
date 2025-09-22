import torch
import torch.utils.checkpoint

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np

class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        
        line = self.data[index]
        line = json.loads(line)
        text = '<s>' + line['text'] + '</s>'
        input_ids = self.tokenizer.encode(text)
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
        input_ids = np.array(input_ids)
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        return {
            'input_ids': torch.from_numpy(X),
            'labels': torch.from_numpy(Y),
        }
        
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
            
    def __len__(self):
        return len(self.data)    

    def __getitem__(self, index):
        line = self.data[index]
        line = json.loads(line)
        
        # 获取对话列表
        conversations = line['conversations']
        messages = []
        # 遍历对话列表
        for conv in conversations:
            # 将每个对话项添加到 messages 列表中
            messages.append({'role': conv['role'], 'content': conv['content']})
        # 应用聊天模板，获取完整的 prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # 获取最后一条对话
        last_conv = conversations[-1]
        
        # 确保最后一条是 assistant 的回复
        if last_conv['role'] != 'assistant':
            # 如果最后一条不是 assistant，则无法确定答案，可以跳过或报错
            raise ValueError("The last conversation must be an assistant's response for SFT.")
        
        # 重新构建 prompt_messages，不包含最后一条 assistant 回复
        prompt_messages = conversations[:-1]
        prompt_str = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        
        # 将最后一条 assistant 回复作为 answer
        answer_str = last_conv['content'] + self.tokenizer.eos_token
        
        # 分词
        prompt_input_ids = self.tokenizer.encode(prompt_str)
        answer_input_ids = self.tokenizer.encode(answer_str)
        
        # 合并 input_ids
        input_ids = prompt_input_ids + answer_input_ids
        
        # 构建 labels
        labels = [0] * len(prompt_input_ids) + answer_input_ids
        
        # 序列长度处理（填充和截断）
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
            labels = labels + [0] * (self.max_seq_len - text_len)

        # 右移处理
        input_ids = input_ids[:-1]
        labels = labels[1:]
        
        return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}

    

class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        # with open(self.data_path, 'r', encoding='utf-8') as f:
        #     self.datas = json.load(f)
        datas = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                datas.append(json.loads(line))
        self.datas = datas
        
    def __getitem__(self, index):
        sample = self.datas[index]
        # 提取 prompt
        prompt = [item['content'] for item in sample['chosen'] if item['role'] == 'user'][0]
        # 提取 chosen 和 rejected
        chosen_response = [item['content'] for item in sample['chosen'] if item['role'] == 'assistant'][0]
        rejected_response = [item['content'] for item in sample['rejected'] if item['role'] == 'assistant'][0]

        # 以下部分与原始代码相同
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(text=text)['input_ids']
        rejected_inputs = self.tokenizer(text=rejected_response)['input_ids'] + [self.tokenizer.eos_token_id]
        chosen_inputs = self.tokenizer(text=chosen_response)['input_ids'] + [self.tokenizer.eos_token_id]
        
        return [prompt_inputs, chosen_inputs, rejected_inputs]
    
    def __len__(self):
        return len(self.datas)
    
    
class DPODataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, features):
        inputs_ids = []
        labels = []
        
        for feature in features:
            inputs_ids.append(feature[0] + feature[1])
            labels.append([0]*len(feature[0]) + feature[1])
        for feature in features:
            inputs_ids.append(feature[0] + feature[2])
            labels.append([0]*len(feature[0]) + feature[2])
            
        def process(inputs_ids, labels):
            inputs_ids = [input_ids[:self.max_seq_len] for input_ids in inputs_ids]
            labels = [label[:self.max_seq_len] for label in labels]
            max_len = max([len(input_ids) for input_ids in inputs_ids])
            batch_input_ids = []
            batch_labels = []
            
            for input_ids, label in zip(inputs_ids, labels):
                if len(input_ids) <= max_len:
                    input_ids = input_ids+[0]*(max_len-len(input_ids))
                    label = label+[0]*(max_len-len(label))
                    batch_input_ids.append(input_ids[:-1])
                    batch_labels.append(label[1:])
            return batch_input_ids, batch_labels
        
        inputs_ids, labels = process(inputs_ids, labels)
        
        return {
            "input_ids": torch.tensor(inputs_ids),
            "labels": torch.tensor(labels)
            }
        
        
            
            
