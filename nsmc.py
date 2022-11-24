# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

args = {
    'train_data_path': './nsmc/ratings_train.txt',
    'val_data_path': './nsmc/ratings_test.txt',
    'save_path': './model',
    'max_epochs': 1,
    'model_path': 'beomi/KcELECTRA-base',
    'batch_size': 32,
    'learning_rate': 5e-5,
    'warmup_ratio': 0.0,
    'max_seq_len': 128
}

df = pd.read_csv(args["train_data_path"], sep='\t')
print(df)



import pandas as pd
import torch

from torch.utils.data import Dataset

class NSMCDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        df = pd.read_csv(csv_file, sep='\t')
        # NaN 값 제거
        df = df.dropna(axis=0)
        # 중복 제거
        df.drop_duplicates(subset=['document'], inplace=True)
        self.input_ids = tokenizer.batch_encode_plus(
            df['document'].to_list(),
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
            truncation=True,
        )['input_ids']
        self.labels = torch.LongTensor(df['label'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
    
    
    
    
from transformers import ElectraForSequenceClassification, ElectraTokenizerFast

model = ElectraForSequenceClassification.from_pretrained(args['model_path'])
tokenizer = ElectraTokenizerFast.from_pretrained(args['model_path'])




from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm

def train(model, train_dataloader, args):
    model.train()
    model.to('cuda')
    global_total_step = len(train_dataloader) * args['max_epochs']
    global_step = 0
    optimizer = AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=global_total_step)
    with tqdm(total=global_total_step, unit='step') as t:
        total = 0
        total_loss = 0
        total_correct = 0
        for epoch in range(args['max_epochs']):
            for batch in train_dataloader:
                global_step += 1
                b_input_ids = batch[0].to('cuda', non_blocking=True)
                b_labels = batch[1].to('cuda', non_blocking=True)
                model.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=b_input_ids,
                    labels=b_labels
                )
                loss, logits = outputs.loss, outputs.logits

                loss.backward()
                optimizer.step()
                scheduler.step()

                preds = logits.detach().argmax(dim=-1).cpu().numpy()
                out_label_ids = b_labels.detach().cpu().numpy()
                total_correct += (preds == out_label_ids).sum()

                batch_loss = loss.item() * len(b_input_ids)

                total += len(b_input_ids)
                total_loss += batch_loss

                t.set_postfix(loss='{:.6f}'.format(batch_loss),
                              accuracy='{:.2f}'.format(total_correct / total * 100))
                t.update(1)
                del b_input_ids
                del outputs
                del loss
                
                
                
                
from torch.utils.data import DataLoader

train_data_set = NSMCDataset(args['train_data_path'], tokenizer, args['max_seq_len'])
train_data_loader = DataLoader(
    dataset=train_data_set,
    batch_size=args['batch_size'],
    pin_memory=True,
    shuffle=True,
    )


train(model, train_data_loader, args)

model.save_pretrained(args['save_path'])







# 평점 10
pos_text = '이방원을 다룬 드라마중 최고였다고 자부함. 진짜 이방원을 보여준 듯이 연기와 인물묘사나 주변상황이 재밌었고 스토리도 진부하지 않았음. 다시 이런드라마를 볼수 있을지~ 진짜 이런 드라마하나 또 나왔음 함.'
# 평점 0
neg_text = '핵노잼 후기보고 낙였네 방금보고왔는데 개실망 재미없어요'

pos_input_vector = tokenizer.encode(pos_text, return_tensors='pt').to('cuda')
pos_pred = model(input_ids=pos_input_vector, labels=None).logits.argmax(dim=-1).tolist()
print(f'{pos_text} : {pos_pred[0]}')

neg_input_vector = tokenizer.encode(neg_text, return_tensors='pt').to('cuda')
neg_pred = model(input_ids=neg_input_vector, labels=None).logits.argmax(dim=-1).tolist()
print(f'{neg_text} : {neg_pred[0]}')