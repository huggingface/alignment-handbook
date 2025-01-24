import os
import json
import torch
import math
import argparse
from tqdm import tqdm
import datasets
from transformers import set_seed
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import hashlib
CE_loss = nn.CrossEntropyLoss()

# torch.multiprocessing.set_start_method('spawn')
set_seed(42)
device = "cuda"
NUM_WORKERS = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None,required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--json_save_path", type=str, default='')
    args = parser.parse_args()
    return args


args = parse_args()
print(args)

from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                            device_map="auto", 
                                            torch_dtype=torch.bfloat16, trust_remote_code=True)
if 'Yi' in args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                           padding_side='right',trust_remote_code=True, 
                                            use_fast=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                            padding_side='right',trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token

model.eval()


class TaskDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.tokenizer = tokenizer
        
        self.data = data

        print(f'total data: {len(self.data)}....')
        self.max_len = args.max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

      data_i = self.data[index]
      messages = data_i['messages']
      input_ids, labels = self.convert_to_chat_format(messages)
      
      
      return {'id': data_i['id'],
              'input_ids':input_ids,
              'labels':labels,
              }
    def convert_to_chat_format(self,messages):
        input_ids = []
        labels = []
        ignore_index = -100
        labels_tokens_num = 0
        if 'Yi' in args.model_name_or_path:
              # "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello, how are you today<|im_end|>\n<|im_start|>user\nJust soso<|im_end|>\n<|im_start|>assistant\n"
              user_tokens=[6,3903,144]
              assistant_tokens=[6, 765, 13611, 144]
              end_tokens=[7, 144]
              for message in messages:
                  role = message["role"]
                  value = message["content"]
                  value_ids = self.tokenizer.encode(value,add_special_tokens=False)

                  if role == "user":
                      input_ids += user_tokens + value_ids + end_tokens
                      labels += [ignore_index] * (len(value_ids)+5)
                  else:
                      input_ids += assistant_tokens + value_ids + end_tokens
                      labels += [ignore_index] * 4 + value_ids + [ignore_index] * 2
              
        elif 'Llama' in args.model_name_or_path:
            # '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a pirate chatbot who always responds in pirate speak!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            input_ids += [128000] #bos
            labels += [ignore_index]
            user_tokens=[128006, 882, 128007, 271]  #<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
            assistant_tokens=[128006, 78191, 128007, 271] #<|start_header_id|>assistant<|end_header_id|>\n\n
            end_tokens=[128009]
            for message in messages:
                role = message["role"]
                value = message["content"]
                value_ids = self.tokenizer.encode(value,add_special_tokens=False)
                if role == "user":
                    input_ids += user_tokens + value_ids + end_tokens
                    labels += [ignore_index] * (len(value_ids)+5)
                else:
                    input_ids += assistant_tokens + value_ids + end_tokens
                    labels += [ignore_index] * 4 + value_ids + [ignore_index] 
                    # labels_tokens_num += len(value_ids)
            # labels = input_ids #testing

        elif 'zephyr' in args.model_name_or_path:
              # <|user|>\nWho are you?</s> \n<|assistant|>\nI am nobody</s> \n<|user|>\nwhy are you herer</s> \n<|assistant|>\nI have no idea</s> \n<|assistant|>\n
              first_user_tokens = [523, 28766,  1838, 28766, 28767, 13]
              user_tokens=[28789, 28766,  1838, 28766, 28767, 13]  #<|user|>\n
              assistant_tokens=[28789, 28766, 489, 11143, 28766, 28767, 13] #<|assistant|>\n
              end_tokens=[2, 28705, 13] #/s> \n
              is_first_turn = True
              for message in messages:
                  role = message["role"]
                  value = '\n' + message["content"]
                  value_ids = self.tokenizer.encode(value,add_special_tokens=False)[2:]
                  if role == "user":
                      if is_first_turn:
                          input_ids += first_user_tokens + value_ids + end_tokens
                          is_first_turn = False
                      else:
                          input_ids += user_tokens + value_ids + end_tokens
                      labels += [ignore_index] * (len(value_ids)+9)
                  else:
                      input_ids += assistant_tokens + value_ids + end_tokens
                      labels += [ignore_index] * 7 + value_ids + [ignore_index]*3
        else:
          raise ValueError(f'no compatible template for {args.model_name_or_path}...')
        
        msg = f'=====path:{args.model_name_or_path}\nori message: [{messages}]\ninput_ids:{input_ids}\ndecode:[{self.tokenizer.decode(input_ids)}]'
        # print(msg)
        # exit()


        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]
        input_ids += [self.tokenizer.pad_token_id] * (
            self.max_len - len(input_ids)
        )
        labels += [ignore_index] * (self.max_len - len(labels))
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        return input_ids, labels

def collate_fn(batch):
    batch_data = {}
    for key in batch[0]:
        batch_data[key] = [example[key] for example in batch]
    # print(batch_data['input_ids'])
    batch_data['input_ids']=nn.utils.rnn.pad_sequence(batch_data['input_ids'], batch_first=True, padding_value=0)
    batch_data['labels']=nn.utils.rnn.pad_sequence(batch_data['labels'], batch_first=True, padding_value=-100)

    
    return batch_data


def compute_loss(all_logits, all_labels, start_token_list=None, end_token_list=None):
    all_losses = []
   
    for j in range(all_logits.size(0)):
        logits = all_logits[j]
        labels = all_labels[j]
        losses = CE_loss(logits[:-1],labels[1:])
        all_losses.append(losses.item())
    
    return all_losses



def main():

    if 'json' in args.data_path:
        ds = datasets.load_dataset('json',data_files=args.data_path)['train']
    else:
        ds = datasets.load_dataset(args.data_path)['train_sft']
    print(ds)

    if 'prompt_id' in ds.column_names:
        ds = ds.rename_column('prompt_id','id')
    print(ds)
    task_dataset = TaskDataset(ds)
    data_loader = torch.utils.data.DataLoader(task_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=NUM_WORKERS, collate_fn=collate_fn,pin_memory=False)
    
    
    
    import time,logging
    logging.basicConfig(filename='out.log', 
                        format   = '%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  
                        datefmt  = '%Y-%m-%d %A %H:%M:%S',
                        level=logging.INFO,
                        filemode = 'w')
    logging.getLogger().setLevel(logging.INFO)
    # logging.info("finish initialize.")
    
    strat_time = time.time()
    all_loss_alone, all_loss_condition, all_logps = [], [], []
    for data in tqdm(data_loader):
        
        with torch.no_grad():
           
            output = model(input_ids=data['input_ids'].to(model.device))  
            logits = output.logits #cpu().numpy()
            
            loss = compute_loss(logits, data['labels'].to(model.device))
            
            all_logps.extend([-l for l in loss])

    ds = ds.add_column(name="logp", column=all_logps)

    ds.to_json(args.json_save_path,force_ascii=False)

    print('Time Used:',(time.time()-strat_time)/60,'(min)')

if __name__ == "__main__":
    main()
