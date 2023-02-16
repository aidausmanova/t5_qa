import time 
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from model.pretrain_t5.triplets_dataset import CustomTripletsDataset

def train(epoch, tokenizer, model, loader, optimizer, writer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"]
        mask = data["source_mask"]

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 50 == 0:
            writer.add_scalar('Loss/train', loss, _)
            # print(f"Epoch {str(epoch)} Loss {str(loss)}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(epoch, tokenizer, model, loader, writer):
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids']
          ids = data['source_ids']
          mask = data['source_mask']

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              print(f'Completed {_}')
            #   writer.add_scalar('Loss/test', loss, _)

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals

def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="./"
):
    training_start_time = time.time()
    local_start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(training_start_time))
    writer = SummaryWriter(output_dir+f"training_logs/trainer_model_{local_start_time_str}")
    torch.manual_seed(model_params["SEED"]) 
    torch.backends.cudnn.deterministic = True

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    dataframe = dataframe[[source_text, target_text]]

    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    val_dataset.to_csv('/export/home/0usmanov/data/conceptnet_val.csv')
    train_dataset = train_dataset.reset_index(drop=True)

    print(f"TRAIN Dataset: {train_dataset.shape}")
    # print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomTripletsDataset(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    # val_set = TripletsDataset(
    #     val_dataset,
    #     tokenizer,
    #     model_params["MAX_SOURCE_TEXT_LENGTH"],
    #     model_params["MAX_TARGET_TEXT_LENGTH"],
    #     source_text,
    #     target_text,
    # )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 4,
    }

    # val_params = {
    #     "batch_size": model_params["VALID_BATCH_SIZE"],
    #     "shuffle": False,
    #     "num_workers": 2,
    # }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    # val_loader = DataLoader(val_set, **val_params)
    print("Created DataLoaders")

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )
    print("Start training")
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, training_loader, optimizer, writer)
        path = os.path.join(output_dir, f"checkpoints/trainer_model_epoch{epoch}_{local_start_time_str}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    print("Finish training")

    # # evaluating test dataset
    # console.log(f"[Initiating Validation]...\n")
    # for epoch in range(model_params["VAL_EPOCHS"]):
    #     predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
    #     final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
    #     final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    # console.save_text(os.path.join(output_dir, "logs.txt"))

    # console.log(f"[Validation Completed.]\n")
    # console.print(
    #     f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    # )
    # console.print(
    #     f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    # )
    # console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")