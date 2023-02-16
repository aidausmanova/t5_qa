import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from model.qa.tmw_dataset import TellMeWhyDataset

class TellMeWhyDataModule(pl.LightningDataModule):
  def __init__(
      self,
      train_df: pd.DataFrame,
      test_df: pd.DataFrame,
      tokenizer: T5Tokenizer,
      batch_size: int = 16,
      source_max_token_len: int = 75,
      target_max_token_len: int = 30
    #   source_max_token_len: int = 396,
    #   target_max_token_len: int = 32
  ):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def setup(self, stage=None):
    self.train_dataset = TellMeWhyDataset(
        self.train_df, 
        self.tokenizer,
        self.source_max_token_len,
        self.target_max_token_len
    )
    
    self.test_dataset = TellMeWhyDataset(
        self.test_df, 
        self.tokenizer,
        self.source_max_token_len,
        self.target_max_token_len
    )

  def train_dataloader(self):
     return DataLoader(
         self.train_dataset,
         batch_size=self.batch_size,
         shuffle=True,
         num_workers=4
     )

  def val_dataloader(self):
     return DataLoader(
         self.test_dataset,
         batch_size=1,
         num_workers=4
     )

  def test_dataloader(self):
     return DataLoader(
         self.test_dataset,
         batch_size=1,
         num_workers=4
     )
