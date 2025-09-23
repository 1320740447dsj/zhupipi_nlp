from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from step_1_pretrain import VLMConfig, VLM


class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                                         {"role": "user", "content": conversations[0]['value']}], \
                                                        tokenize=False, \
                                                        add_generation_prompt=True).replace('<image>',
                                                                                            '<|image_pad|>' * self.config.image_pad_num)
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                                         {"role": "user", "content": "图片内容是什么\n<image>"}], \
                                                        tokenize=False, \
                                                        add_generation_prompt=True).replace('<image>',
                                                                                            '<|image_pad|>' * self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }


def find_assistant_tokens(tokenizer, target):
    result = []
    start_index =0
    end_index = 0
    while start_index <= len(target)-1:
        if target[start_index]!=tokenizer('assistant')['input_ids'][0]:
            start_index+=1
            end_index+=1
        else:
            end_index+=1
            if target[end_index]==tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index+1,end_index+1))
                start_index=end_index+1
    return result



class SFTDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        # 逐行读取 jsonl 文件
        self.datas = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.datas.append(json.loads(line))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            messages = [{"role": "system", "content": 'You are a helpful assistant.'}]
            for conversation in conversations:
                if conversation['role'] == 'user':
                    messages.append({"role": "user", "content": conversation['content']})
                else:
                    messages.append({"role": "assistant", "content": conversation['content']})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False).replace('<image>',
                                                                                        '<|image_pad|>' * self.config.image_pad_num)
            input_ids = self.tokenizer(text)['input_ids']
            indexs = find_assistant_tokens(self.tokenizer, input_ids)
            labels = len(input_ids) * [self.tokenizer.pad_token_id]
            for index in indexs:
                labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
            input_ids = input_ids[:-1]
            labels = labels[1:]

            image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
            pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(images=default_image, return_tensors="pt")["pixel_values"]
            q_text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": 'You are a helpful assistant.'},
                 {"role": "user", "content": "图片内容是什么\n<image>"}],
                tokenize=False,
                add_generation_prompt=True
            ).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }