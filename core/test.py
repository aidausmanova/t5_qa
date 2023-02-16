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


if __name__ == "__main__":
    model_dir = '/export/home/0usmanov/project/output/code_encoder/checkpoints/t5s_model_2023-02-10_11-25-04/'
    # model = 't5s_model_2023-02-10_11-25-04/'
    model_dir = '/export/home/0usmanov/project/output/code_encoder/training_logs/checkpoint-11000/'

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained(model_dir + 'pytorch_model.bin', local_files_only=True, config=model_dir + 'config.json')

    # sent = "<extra_id_0> is capable of bark"
    # output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    # print(tokenizer.decode(output_ids[0]))

    # sent = "dogs and <extra_id_0> are pets"
    # output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    # print(tokenizer.decode(output_ids[0]))

    # sent = "baseball is a <extra_id_0>"
    # output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    # print(tokenizer.decode(output_ids[0]))

    # sent = "<extra_id_0> eats banana"
    # output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    # print(tokenizer.decode(output_ids[0]))

    # sent = "students go to <extra_id_0> to study"
    # output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    # print(tokenizer.decode(output_ids[0]))

    sent = "Harry and Ron are <extra_id_0>"
    output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    print(tokenizer.decode(output_ids[0]))

    sent = "the movie was <extra_id_0>, I did not like it"
    output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    print(tokenizer.decode(output_ids[0]))

    sent = "snakes are <extra_id_0> for human"
    output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    print(tokenizer.decode(output_ids[0]))

    sent = "people <extra_id_0> at jokes"
    output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    print(tokenizer.decode(output_ids[0]))

    sent = "the water in the river is <extra_id_0>"
    output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    print(tokenizer.decode(output_ids[0]))

    sent = "<extra_id_0> is prerequisite for running"
    output_ids = model.generate(tokenizer.encode(sent, return_tensors='pt'))
    print(tokenizer.decode(output_ids[0]))