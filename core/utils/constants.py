from os.path import dirname, join
from pathlib import Path

PRETRAIN_PARAMETERS = {
    "MODEL": 't5-small',
    "TOKENIZER": 't5-small',
    "LEARNING_RATE": 1e-3,
    "TRAIN_EPOCHS": 3,
    "VAL_EPOCHS": 1,
    "TRAIN_BATCH_SIZE": 10,
    "VALID_BATCH_SIZE": 1,
    "MAX_SOURCE_TEXT_LENGTH": 512,
    "MAX_TARGET_TEXT_LENGTH": 50,
    "SEED": 42
}

FINETUNE_PARAMETERS = {
    "MODEL": 't5-small',
    "TOKENIZER": 't5-small',
    "LEARNING_RATE": 0.0001,
    "TRAIN_EPOCHS": 3,
    "VAL_EPOCHS": 1,
    "TRAIN_BATCH_SIZE": 8,
    "VALID_BATCH_SIZE": 1,
    "MAX_SOURCE_TEXT_LENGTH": 396,
    "MAX_TARGET_TEXT_LENGTH": 32,
    "SEED": 42
}

PROJECT_ROOT_DIR = dirname(Path(__file__).resolve().parents[2])