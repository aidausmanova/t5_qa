import pandas as pd
from transformers import T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class ConceptnetDataset(Dataset):
  def __init__(self, tokenizer, dataset, max_len=512):

    self.data = dataset
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.tokenizer.max_length = max_len
    self.tokenizer.model_max_length = max_len
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    print("In dataset")
    source_text = str(self.data['input'][index])
    target_text = str(self.data['target'][index])

    input_ = source_text.lower() + ' </s>'
    target = target_text.lower() + ' </s>'

    # tokenize inputs
    tokenized_inputs = self.tokenizer.batch_encode_plus(
        [input_], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
    )
    # tokenize targets
    tokenized_targets = self.tokenizer.batch_encode_plus(
        [target],max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
    )

    source_ids = tokenized_inputs["input_ids"].squeeze()
    target_ids = tokenized_targets["input_ids"].squeeze()

    src_mask    = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
    target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


class CustomTripletsDataset(Dataset):
  def __init__(
      self, 
      dataframe, 
      tokenizer, 
      source_len, 
      target_len, 
      source_text, 
      target_text):
    
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.summ_len = target_len
    self.target_text = self.data[target_text]
    self.source_text = self.data[source_text]

  def __len__(self):
    return len(self.target_text)

  def __getitem__(self, index):
    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }


class TripletsDataset(Dataset):
  def __init__(
      self,
      data: pd.DataFrame,
      tokenizer: T5Tokenizer,
      source_max_token_len: int = 512,
      target_max_token_len: int = 512
  ):

    self.tokenizer = tokenizer
    self.data = data
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    # input = 'ConceptNet: '+data_row['input']+'</s>' 
    # target = data_row['target']+'</s>'  
    # input = 'ConceptNet: '+data_row['input']
    input = data_row['input']
    target = data_row['target']  

    source_encoding = self.tokenizer(
        input,
        max_length=self.source_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    target_encoding = self.tokenizer(
        target,
        max_length=self.target_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    labels = target_encoding["input_ids"]
    labels[labels==0] = -100

    return dict(
        input=input,
        target=target,
        input_ids=source_encoding["input_ids"].flatten(),
        attention_mask=source_encoding["attention_mask"].flatten(),
        labels=labels.flatten(),
        labels_attention_mask=target_encoding["attention_mask"].flatten()
    )
