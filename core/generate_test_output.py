import sys
sys.path.insert(1, '/export/home/0usmanov/project/src/core')
import random
import argparse
import torch

from carbontracker.tracker import CarbonTracker
from datasets import load_dataset
from nvitop import Device
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.qa.tmw_data_module import TellMeWhyDataModule
from model.qa import qamodel
from model.pretrain_t5.pretrained_model import Pretrained_T5, T5Pretrainer
from model.pretrain_t5 import triplets_data_module
from model.pretrain_t5.trainer import T5Trainer
from utils.services import extract_questions_answers, generate_opposite_mask
from utils.constants import PRETRAIN_PARAMETERS, FINETUNE_PARAMETERS


def extract_questions_answers(data):
  data_rows = []

  for element in data:
    context = element['narrative']
    question = element['question']
    answer = element['answer']
    is_question_answerable = element['is_ques_answerable']
    meta = element['question_meta']

    data_rows.append({
        "question": question,
        "gold_answer": answer,
        "narrative": context,
        "meta": meta,
        "is_ques_answerable": is_question_answerable
    })
  return pd.DataFrame(data_rows)

def generate_answer(model, tokenizer, question, context):
    source_encoding = tokenizer(
        question,
        context,
        max_length=396,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = model.model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask=source_encoding["attention_mask"],
        num_beams=1,
        max_length=80,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = [
        tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    ]

    return "".join(preds)

if __name__ == "__main__":
    t5_model = "t5-small"
    root_path = "/export/home/0usmanov/project/output/"
    load_model = "t5s_weighted_2023-01-22_13-23-27"
    tokenizer = T5Tokenizer.from_pretrained(t5_model)

    print("Preparing test data ...")
    tellmewhy = load_dataset('StonyBrookNLP/tellmewhy')
    test_data = tellmewhy['test']
    test_df = extract_questions_answers(tellmewhy['test'])
    # train_dataset = TellMeWhyDataset(test_df, tokenizer, 396, 32)
    # test_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)

    data_rows = []

    print("Loading the model ...")
    qa_model = qamodel.QAModel.load_from_checkpoint(f"{root_path}tellmewhy/checkpoints/{load_model}.ckpt")
    qa_model.freeze()
    # qa_model.model.eval()

    print("Generating predictions ...")
    for test_sample in tqdm(test_data):
        data_rows.append({
            "question": test_sample['question'],
            "gold_answer": test_sample['answer'],
            "narrative": test_sample['narrative'],
            "meta": test_sample['question_meta'],
            "is_ques_answerable": test_sample['is_ques_answerable'],
            "predicted_answer": generate_answer(qa_model, tokenizer, test_sample['question'], test_sample['narrative'])
        })
        # targets.append(test_sample['answer'])
        # predictions.append(generate_answer(qa_model, tokenizer, test_sample))

    
    finish_time = time.time()
    local_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(finish_time))
    
    output_df = pd.DataFrame(data_rows)
    output_df.to_csv(f"{root_path}test_output/{load_model}.csv")
    print(f"Finished at {local_time_str}")

    # zipped = list(zip(targets, predictions))
    # output_df = pd.DataFrame(zipped, columns=['target', 'prediction'])
    # output_df.to_csv(f"{root_path}test_output/{load_model}.csv")

    # for batch in tqdm(test_loader):
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     outs = qa_model.model.generate(input_ids=input_ids, 
    #                                     attention_mask=attention_mask,
    #                                     num_beams=1,
    #                                     max_length=80,
    #                                     repetition_penalty=2.5,
    #                                     length_penalty=1.0,
    #                                     early_stopping=True,
    #                                     use_cache=True)
    #     preds = [
    #         tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #         for generated_id in outs
    #     ]

    