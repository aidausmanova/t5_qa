import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from model.pretrain_t5.triplets_dataset import ConceptnetDataset
# from .utils.services import get_dataset


# New class (used currently)
class T5Pretrainer(pl.LightningModule):
    def __init__(self, train_data, val_data, hparam):
        super(T5Pretrainer, self).__init__()
        self.hparam = hparam

        self.model = T5ForConditionalGeneration.from_pretrained(
            hparam.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparam.model_name_or_path
        )
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.save_hyperparameters()
        self.tokenizer.max_length = self.hparam.max_seq_length
        self.tokenizer.model_max_length = self.hparam.max_seq_length
        print("In model")

    def is_logger(self):
        return True

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
      loss = self._step(batch)

      # tensorboard_logs = {"train_loss": loss}
      self.log("train_loss", loss, prog_bar=True, logger=True)
      return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        perplexity = torch.exp(avg_train_loss)
        # tensorboard_logs = {"avg_train_loss": avg_train_loss, "train_perplexity": torch.exp(avg_train_loss)}
        self.log("avg_train_loss", avg_train_loss, prog_bar=True, logger=True)
        self.log("train_perplexity", perplexity, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        perplexity = torch.exp(avg_loss)
        # tensorboard_logs = {"val_loss": avg_loss, "val_perplexity": torch.exp(avg_loss)}
        self.log("val_loss", avg_loss, prog_bar=True, logger=True)
        self.log("val_perplexity", perplexity, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparam.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparam.learning_rate, eps=self.hparam.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        # train_dataset = get_dataset(
        #     tokenizer=self.tokenizer, type_path="train", args=self.hparam)
        
        print("Train set: ", self.train_dataset.shape)
        train_data = ConceptnetDataset(tokenizer=self.tokenizer, dataset=self.train_dataset)
        print("Train data: ", train_data.__len__())
        dataloader = DataLoader(
            train_data, batch_size=self.hparam.train_batch_size,
            drop_last=True, shuffle=True, num_workers=2)
        
        t_total = (
            (len(dataloader.dataset) //
             (self.hparam.train_batch_size * max(1, self.hparam.n_gpu)))
            // self.hparam.gradient_accumulation_steps
            * float(self.hparam.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        # val_dataset = get_dataset(
        #     tokenizer=self.tokenizer, type_path="validation", args=self.hparam)
        print("Val set: ", self.val_dataset.shape)
        val_data = ConceptnetDataset(tokenizer=self.tokenizer, dataset=self.val_dataset)
        return DataLoader(val_data, batch_size=self.hparam.eval_batch_size, num_workers=2)



class Pretrained_T5(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
  
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

  def configure_optimizers(self):
     return AdamW(self.parameters(), lr=1e-3)
