import sys
sys.path.insert(1, '/export/home/0usmanov/project/src/core')
import random
import argparse
import torch
import time
import torch

from comet_ml import Experiment
from datasets import load_dataset
from datetime import timedelta
from nvitop import Device
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Set
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Checkpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loops import FitLoop
# from lightning_fabric.utilities.types import _PATH
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

from carbontracker.tracker import CarbonTracker
from carbontracker import parser

from model.qa.tmw_data_module import TellMeWhyDataModule
from model.qa import qamodel
from model.pretrain_t5.pretrained_model import Pretrained_T5, T5Pretrainer
from model.pretrain_t5 import triplets_data_module
from model.pretrain_t5.trainer import T5Trainer
from utils.services import extract_questions_answers, generate_opposite_mask
from utils.constants import PRETRAIN_PARAMETERS, FINETUNE_PARAMETERS


args_dict = dict(
    root_path = "/export/home/0usmanov/project/output/",
    output_dir="", # path to save the checkpoints
    model_variant="t5s",
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=512,
    learning_rate=5e-5,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=16,
    eval_batch_size=16,
    num_train_epochs=30,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
)

def train_pl_conceptnet_model():
  df = pd.read_csv('/export/home/0usmanov/data/masked_input_target.csv')
  pl.seed_everything(42)
  tokenizer = T5Tokenizer.from_pretrained(PRETRAIN_PARAMETERS['MODEL'])

  train_size = 0.8
  train_dataset = df.sample(frac=train_size, random_state=PRETRAIN_PARAMETERS["SEED"])
  val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
  train_dataset = train_dataset.reset_index(drop=True)

  print("Created DataModule")
  triplets_data_module = triplets_data_module.TripletsDataModule(train_dataset, val_dataset, tokenizer, batch_size=PRETRAIN_PARAMETERS["TRAIN_BATCH_SIZE"])
  triplets_data_module.setup()

  pretrained_t5_model = Pretrained_T5()

  # tracker = CarbonTracker(epochs=10)

  training_start_time = time.time()
  print("Start training " + local_start_time_str)
  # Save best checkopoint
  local_start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(training_start_time))
  checkpoint_callback = ModelCheckpoint(
      dirpath="/export/home/0usmanov/project/output/conceptnet/checkpoints",
      filename=f"pl_model_{local_start_time_str}",
      save_top_k=1,
      every_n_epochs=5, # check the model after every epoch
      verbose=True,
      monitor="val_loss",
      mode="min"
  )

  logger = TensorBoardLogger("/export/home/0usmanov/project/output/conceptnet/training_logs", name=f"pl_model_{local_start_time_str}")
  trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        callbacks=checkpoint_callback,
        gpus=1,
        enable_progress_bar=True,
        max_epochs=20
        # max_epochs=PRETRAIN_PARAMETERS["train_epochs"]
    )

  # Train the model
  trainer.fit(pretrained_t5_model, triplets_data_module)
  print("Finished")

class TrainLoop(FitLoop):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def advance(self):
        """Advance from one iteration to the next."""
        print("EPOCH START")
        self.tracker.epoch_start()

    def on_advance_end(self):
        """Do something at the end of an iteration."""
        print("EPOCH END")
        self.tracker.epoch_end()

    def on_run_end(self):
        """Do something when the loop ends."""


class MyCheckpoint(ModelCheckpoint):
  def __init__(self, 
        dirpath,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None, 
        tracker: Optional[CarbonTracker] = None):
    super(MyCheckpoint, self).__init__(dirpath,filename,monitor,verbose,save_last,save_top_k,save_weights_only,mode,auto_insert_metric_name,
                                 every_n_train_steps,train_time_interval,every_n_epochs,save_on_train_epoch_end)
    self.tracker = CarbonTracker(epochs=1)

  def on_train_epoch_start(self, *arg, **kwargs):
    print("EPOCH START")
    self.tracker.epoch_start()

  def on_train_epoch_end(self, *arg, **kwargs):
    print("EPOCH END")
    self.tracker.epoch_end()

  def on_train_end(self, *arg, **kwargs):
    self.tracker.stop()
    # logs = parser.parse_all_logs(log_dir="/export/home/0usmanov/project/output/carbontracker/")
    # first_log = logs[0]

    # print(f"Output file name: {first_log['output_filename']}")
    # print(f"Standard file name: {first_log['standard_filename']}")
    # print(f"Stopped early: {first_log['early_stop']}")
    # print(f"Measured consumption: {first_log['actual']}")
    # print(f"Predicted consumption: {first_log['pred']}")
    # print(f"Measured GPU devices: {first_log['components']['gpu']['devices']}")




if __name__ == "__main__":
    args = argparse.Namespace(**args_dict)
    experiment = Experiment(api_key="ggL2AArbgC6Ve3j7Ww3xMLLMK")

    pl.seed_everything(42)
    args = argparse.Namespace(**args_dict)
    
    exp_type = "nopretrain" # weighted, random, new, nopretrain
    root_path = "/export/home/0usmanov/project/output/"
    load_model = "t5s_weighted_pretrainer_model_2023-01-05_08-56-25-v4.ckpt"
    model_path = f"{root_path}conceptnet/checkpoints/{load_model}"
    tellmewhy = load_dataset('StonyBrookNLP/tellmewhy')

    print("Loaded dataset")
    tmw_df = extract_questions_answers(tellmewhy["train"])
    tmw_df = tmw_df.sample(frac = 1)
    val_tmw_df = extract_questions_answers(tellmewhy["validation"])

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    data_module = TellMeWhyDataModule(tmw_df, val_tmw_df, tokenizer, batch_size=args.train_batch_size)
    data_module.setup()

    print("Create QA model")
    # qa_model = qamodel.QAModel.load_from_checkpoint(model_path)
    qa_model = qamodel.QAModel()

    training_start_time = time.time()
    local_start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(training_start_time))
    print("Start training " + local_start_time_str)

    # Save best checkopoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_path}tellmewhy/checkpoints",
        # filename=f"{args.model_variant}_{exp_type}_bs{args.train_batch_size}_{local_start_time_str}",
        filename=f"{args.model_variant}_{exp_type}_bs{args.train_batch_size}_{local_start_time_str}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    # custom_checkpoint_callback = MyCheckpoint(
    #     dirpath=f"{root_path}tellmewhy/checkpoints",
    #     filename=f"{args.model_variant}_{exp_type}_{local_start_time_str}",
    #     save_top_k=1,
    #     verbose=True,
    #     monitor="val_loss",
    #     mode="min"
    # )

    # logger = TensorBoardLogger(f"{root_path}tellmewhy/training_logs", name=f"{args.model_variant}_{exp_type}_bs{args.train_batch_size}_{local_start_time_str}")
    logger = TensorBoardLogger(f"{root_path}tellmewhy/training_logs", name=f"{args.model_variant}_{exp_type}_bs{args.train_batch_size}_{local_start_time_str}")

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        callbacks=checkpoint_callback,
        gpus=args.n_gpu,
        enable_progress_bar=True,
        max_epochs=args.num_train_epochs,
        accumulate_grad_batches=args.gradient_accumulation_steps
    )

    print("Start training")
    # trainer.fit_loop = TrainLoop(tracker)
    trainer.fit(qa_model, data_module)
    print("Training finished")
