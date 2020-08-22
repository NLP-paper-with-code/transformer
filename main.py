from data_loader.data_loader import DataLoader
from model.transformer import Transformer
from constants.constants import Constants
from train import Trainer

import torch.optim as optim
import numpy as np
import torch

import random

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  const = Constants()
  data_loader = DataLoader(device=device)
  
  SOURCE_VOCAB_SIZE = data_loader.get_source_vocab_size()
  TARGET_VOCAB_SIZE = data_loader.get_target_vocab_size()
  SRC_PAD_IDX = data_loader.get_source_padding_index()
  TRG_PAD_IDX = data_loader.get_target_padding_index()

  train_iterator, valid_iterator, test_iterator = data_loader.get_iterator()

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
    device=device
  ).to(device)

  model.apply(model_utils.initialize_weights)

  optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE)

  trainer = Trainer(
    const=const,
    model=model,
    optimizer=optimizer,
    train_iterator=train_iterator,
    valid_iterator=valid_iterator,
    device=device
  )

  trainer.train()