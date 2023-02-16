import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, T5Config

model_dir = '/export/home/0usmanov/project/output/code_encoder/training_logs/checkpoint-11000/'


class QAModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
   #  self.model = T5ForConditionalGeneration.from_pretrained(model_dir + 'pytorch_model.bin', local_files_only=True, config=model_dir + 'config.json')
   #  self.model.load_state_dict(torch.load('/content/drive/MyDrive/Study/Masters/Thesis/t5model_1.pt'))
   #  self.config = T5Config.from_pretrained("t5-small")
   #  self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, config=self.config)
   
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask = batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self.forward(input_ids, attention_mask, labels)
     self.log("train_loss", loss, prog_bar=True, logger=True)
     return loss
  
  def validation_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask = batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self.forward(input_ids, attention_mask, labels)
     self.log("val_loss", loss, prog_bar=True, logger=True)
     return loss

  def test_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask = batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self.forward(input_ids, attention_mask, labels)
     self.log("test_loss", loss, prog_bar=True, logger=True)
     return loss

   # def training_epoch_end(self):
   #    self.tracker.epoch_end()
   #    self.tracker.epoch_start()

  def configure_optimizers(self):
     return AdamW(self.parameters(), lr=5e-5)
