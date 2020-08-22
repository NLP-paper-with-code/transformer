import utility.tokenizer as tokenizer

from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from torchtext.data import Dataset

import torchtext

from typing import Tuple

class DataLoader(object):
  def __init__(self, device):
    self.source = Field(tokenize = tokenizer.tokenize_de,
        init_token = '<sos>',
        eos_token = '<eos>',
        lower = True,
        batch_first = True)

    self.target = Field(tokenize = tokenizer.tokenize_en,
        init_token = '<sos>',
        eos_token = '<eos>',
        lower = True,
        batch_first = True)

    train_data, valid_data, test_data = self.split_data()
    self.build_vocab(train_data=train_data)
    self.device = device

  def get_source_vocab_size() -> int:
    return len(source.vocab)

  def get_target_vocab_size() -> int:
    return len(target.vocab)

  def get_source_padding_index() -> int:
    return self.source.stoi[source.pad_token]
  
  def get_target_padding_index() -> int:
    return self.target.stoi[target.pad_token]

  def build_vocab(self, train_data: Dataset):
    source.build_vocab(self.train_data, min_freq = 2)
    target.build_vocab(self.train_data, min_freq = 2)
  
  def split_data(self) -> Tuple[Dataset]:
    return Multi30k.splits(
      exts=('.de','.en'),
      fields=(self.source, self.target)
    )

  def get_iterator(
    self,
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset):
    return BucketIterator.splits(
      (train_data, valid_data, test_data),
      batch_size=const.BATCH_SIZE,
      device=device)

