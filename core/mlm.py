import argparse
import tarfile
import time
import gzip
import re
import io
from io import TextIOWrapper
import json
import ast
from urllib.request import urlopen
import urllib
import random
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

import torch
from tqdm.notebook import tqdm
import time
import warnings
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from transformers import (
    BatchEncoding,
    DataCollator,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TFAutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    set_seed
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Any, Callable, Dict, List, NewType, Tuple, Union

from comet_ml import Experiment

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

args_dict = dict(
    train_url='https://cloud.tsinghua.edu.cn/d/670f7787b6554f308226/files/?p=%2Fcommonsense_data%2Ftrain.txt&dl=1',
    val_url='https://cloud.tsinghua.edu.cn/d/670f7787b6554f308226/files/?p=%2Fcommonsense_data%2Fvalid.txt&dl=1',
    test_url='https://cloud.tsinghua.edu.cn/d/670f7787b6554f308226/files/?p=%2Fcommonsense_data%2Ftest.txt&dl=1', 
    output_dir="", # path to save the checkpoints
    model_variant="t5s",
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=512,
    learning_rate=5e-5,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=32,
    eval_batch_size=32,
    num_train_epochs=5,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)


def get_data(path):
  response = urllib.request.urlopen(path)
  lines = TextIOWrapper(response, encoding='utf-8')
  return pd.DataFrame({'text':lines})

def get_tokens(df):
  tokens = []
  for line in df.text.to_list():
    tokens.append(" ".join([word for word in line.split() if len(word) > 3]))
  return tokens

def get_random_token(df):
  token = []
  for line in df.tokens.to_list():
    token.append(random.choice(line.split()))
  return token


class MethodDataset(Dataset):
    def __init__(
        self, tokenizer, df, mode, block_size: int = 256, overwrite_cache=False, local_rank=-1,
    ):
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        self.examples = []
        for text, token in tqdm(list(zip(df.text.values, df.token.values))):
            inpt = f'{text}'
            outpt = f'<{token}> {text}'
            input_encodings = tokenizer.encode_plus(inpt, pad_to_max_length = True, max_length = block_size, truncation = True)
            target_encodings = tokenizer.encode_plus(outpt, pad_to_max_length = True, max_length = block_size, truncation = True)

            encodings = {
                'input_ids': torch.tensor(input_encodings['input_ids'] + [tokenizer.eos_token_id], dtype = torch.long), 
                'attention_mask': torch.tensor(input_encodings['attention_mask'] + [tokenizer.eos_token_id], dtype = torch.long),
                'target_ids': torch.tensor(target_encodings['input_ids'] + [tokenizer.eos_token_id], dtype = torch.long),
                'target_attention_mask': torch.tensor(target_encodings['attention_mask'] + [tokenizer.pad_token_id], dtype = torch.long)
            }
            
            self.examples.append(encodings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return self.examples[i]

@dataclass
class DataCollatorForSeq2SeqMaskLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _noise_span_to_unique_sentinel(self, tokens, mask, max_sentinels, sentinel_id):
        sentineled_toks = tokens.clone()
        prev_tok_noise = torch.nn.functional.pad(mask[:-1], [1, 0])

        first_noise_toks = torch.logical_and(mask, ~prev_tok_noise)
        subse_noise_toks = torch.logical_and(mask, prev_tok_noise)
        
        sentinels = torch.arange(start = sentinel_id, end = sentinel_id - max_sentinels, step = -1)
        sentineled_toks[first_noise_toks] = sentinels[:first_noise_toks.sum().item()]
        return sentineled_toks[~subse_noise_toks]

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability = 0.15, min_span_length = 1, max_span_length = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(inputs)
        device = inputs.device
        inpts = inputs.clone()
        span_lengths = torch.randint(low = min_span_length, high = max_span_length + 1, size = (inpts.shape[0],), device = device)
        periods = torch.round(span_lengths / mlm_probability)
        offsets = torch.tensor([random.randint(0, period.item()) for period in periods], device = device)
        masks = torch.stack([(torch.arange(start = 0, end = inpts.shape[1]) + offset) % period < span for offset, period, span in zip(offsets, periods, span_lengths)])

        if self.tokenizer._pad_token is not None:
            padding_mask = inpts.eq(self.tokenizer.pad_token_id)
            masks.masked_fill_(padding_mask, value = False)
        num_masks = torch.floor_divide(masks.sum(axis = 1), span_lengths)
        new_inpts = []
        lbls = []
        for inpt, mask in zip(inpts, masks):
            new_inpts.append(
                self._noise_span_to_unique_sentinel(inpt, mask, 100, tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
            )
            lbls.append(
                self._noise_span_to_unique_sentinel(inpt, ~mask, 100, tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
            )

        new_inpts = pad_sequence(new_inpts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        lbls = pad_sequence(lbls, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return new_inpts, lbls


if __name__ == "__main__":
    args = argparse.Namespace(**args_dict)
    experiment = Experiment(api_key="ggL2AArbgC6Ve3j7Ww3xMLLMK")


    print("Start data loading")
    train_df = get_data(args.train_url)
    train_df['text'] = train_df['text'].str.replace('\n', '', regex=True)
    val_df = get_data(args.val_url)
    val_df['text'] = val_df['text'].str.replace('\n', '', regex=True)
    print("Shapes: ", train_df.shape, val_df.shape)

    print("Preparing data")
    train_df['tokens'] = get_tokens(train_df)
    val_df['tokens'] = get_tokens(val_df)

    # train_df['tokens'].replace(' ', np.nan, inplace=True)
    # train_df.dropna(subset=['tokens'], inplace=True)
    train_df = train_df[train_df.tokens != '']
    val_df = val_df[val_df.tokens != '']
    print(len(train_df[train_df.tokens == '']))

    train_df['token'] = get_random_token(train_df)
    val_df['token'] = get_random_token(val_df)

    print("Select data fraction")
    sample = 0.1
    # trn_df = train_df.sample(frac = sample)
    # val_df = val_df.sample(frac = sample)

    print("Create dataset")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenizer.add_special_tokens({'mask_token': '<mask>'})
    train_dataset = MethodDataset(tokenizer, train_df, 'train')
    valid_dataset = MethodDataset(tokenizer, val_df, 'valid')

    print("Load model")
    training_start_time = time.time()
    local_start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(training_start_time))
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="/export/home/0usmanov/project/output/conceptnet/checkpoints", 
    #     filename=f"mlm_model_{local_start_time_str}", 
    #     monitor="val_loss", mode="min", save_top_k=1
    # )
    # logger = TensorBoardLogger("/export/home/0usmanov/project/output/conceptnet/training_logs", name=f"mlm_model_{local_start_time_str}")


    training_args = TrainingArguments(
        output_dir="/export/home/0usmanov/project/output/code_encoder/training_logs",
        overwrite_output_dir=True,
        num_train_epochs=10,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        fp16=True,
        weight_decay=args.weight_decay, 
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps = 1000,
        # logging_dir='/export/home/0usmanov/project/output/code_encoder/training_logs',
        logging_steps=1000, 
        save_total_limit=1,
        do_train = True,
        do_eval = True,
        load_best_model_at_end= True

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2SeqMaskLanguageModeling(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    print("Start training")
    trainer.train()
    print("Start evaluation")
    trainer.evaluate()
    print("Finished")
    model_dir = '/export/home/0usmanov/project/output/code_encoder/checkpoints/'
    trainer.save_model(model_dir + f"t5s_{local_start_time_str}")

    # alternative saving method and folder
    # model.save_pretrained(model_dir + f"t5s_model_{local_start_time_str}") 
