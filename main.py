from data_loader.data_loader import DataLoader
from model.transformer import Transformer
from constants.constants import Constants
from train import Trainer

import utility.inference as inference_utils
import utility.model as model_utils

import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import spacy

from pathlib import Path

import random
import os

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  Path(f'{os.getcwd()}/.data').mkdir(parents=True, exist_ok=True)
  Path(f'{os.getcwd()}/checkpoints').mkdir(parents=True, exist_ok=True)
  
  const = Constants()
  data_loader = DataLoader(device=device, const=const)

  source, target = data_loader.get_fields()
  train_data, valid_data, test_data = data_loader.split_data(source=source, target=target)
  train_iterator, valid_iterator, test_iterator = data_loader.get_iterator(
    train_data=train_data,
    valid_data=valid_data,
    test_data=test_data,
  )
  
  SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE = data_loader.get_vocab_size(
    source=source,
    target=target,
  )

  SRC_PAD_IDX, TRG_PAD_IDX = data_loader.get_padding_index(
    source=source,
    target=target,
  )

  random.seed(const.SEED)
  np.random.seed(const.SEED)
  torch.manual_seed(const.SEED)
  torch.cuda.manual_seed(const.SEED)
  torch.backends.cudnn.deterministic = True

  model = Transformer(
    source_vocab_size=SOURCE_VOCAB_SIZE,
    target_vocab_size=TARGET_VOCAB_SIZE,
    source_padding_index=SRC_PAD_IDX,
    target_padding_index=TRG_PAD_IDX,
    embedding_size=const.EMBEDDING_SIZE,
    number_of_layers=const.NUMBER_OF_LAYERS,
    number_of_heads=const.NUMBER_OF_HEADS,
    forward_expansion=const.FORWARD_EXPANSION,
    device=device,
  ).to(device)

  model.apply(model_utils.initialize_weights)
  optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
  cross_entropy = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

  print(f'The model has {model_utils.count_parameters(model):,} trainable parameters')

  trainer = Trainer(
    const=const,
    optimizer=optimizer,
    criterion=cross_entropy,
    device=device,
  )

  trainer.train(
    model=model,
    train_iterator=train_iterator,
    valid_iterator=valid_iterator,
  )

  model.load_state_dict(torch.load('./checkpoints/model.best.pt'))
  
  trainer.test(model=model, test_iterator=test_iterator)

  bleu_score = inference_utils.calculate_bleu(
    data=test_data,
    source_field=source,
    target_field=target,
    model=model,
    device=device,
  )

  print(f'BLEU score = {bleu_score*100:.2f}')