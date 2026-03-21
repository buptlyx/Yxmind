from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    #init
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        #输入给GPU的最大长度
        self.max_length = max_length
        #使用huggingface的dataset库加载数据集
        self.samples = load_dataset('json', data_files=data_path, split='train')
    #__len__
    def __len__(self):
        return len(self.samples)
    #__getitem__
    #我们拿到的是，jsonl里的每一行数据
    def __getitem__(self, idx):
        #取出一个具体的数据
        sample = self.samples[idx]
    #tokenizer把文本转换为input_ids
        tokens=self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length-2,#留出位置给BOS和EOS
            truncation=True,#如果长度超过了max，自动剪切
        ).input_ids
    #我们需要加上<BOS>和<EOS>，以及<PAD>填充
        tokens=[self.tokenizer.bos_token_id]+tokens+[self.tokenizer.eos_token_id]
        #填充到max_length
        input_ids=tokens+[self.tokenizer.pad_token_id]*(self.max_length-len(tokens))
        #把input_ids转换为tensor
        input_ids=torch.tensor(input_ids,dtype=torch.long)
    #需要自行编写labels，防止<PAD>参与loss计算
        labels=input_ids.clone()
        #把PAD位置的标签设置为-100，告诉模型这些位置不参与loss计算
        labels[labels==self.tokenizer.pad_token_id]=-100
    #需要编写attention_mask，告诉模型哪些位置是有效的，哪些位置是<PAD>
        #非PAD的位置为1，PAD的位置为0
        attention_mask=(input_ids!=self.tokenizer.pad_token_id).long()
    #我们要输出的是input_ids，labels，attention_mask
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }