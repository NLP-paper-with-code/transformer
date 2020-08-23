from constants.constants import Constants

import utility.tokenizer as tokenizer

from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from torchtext.data import Dataset

import torchtext
import dill

from typing import Tuple
from pathlib import Path

class DataLoader(object):
  def __init__(self, device, const: Constants):
    self.device = device
    self.const = const

  def get_vocab_size(self, source: Field, target: Field) -> Tuple[int]:
    return len(source.vocab), len(target.vocab)
  
  def get_padding_index(self, source: Field, target: Field) -> Tuple[int]:
    return (
      source.vocab.stoi[source.pad_token],
      target.vocab.stoi[target.pad_token],
    )

  def get_fields(self) -> Tuple[Field]:
    source, target = None, None
    if Path('./.data/source.Field').is_file():
      with open('./.data/source.Field', 'rb') as source_file, \
        open('./.data/target.Field', 'rb') as target_file:

        source = dill.load(source_file)
        target = dill.load(target_file)
    
    else:
      source = Field(
        tokenize=tokenizer.tokenize_de,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=True)

      target = Field(
        tokenize=tokenizer.tokenize_en,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=True)

    return source, target

  def split_data(self, source: Field, target: Field) -> Tuple[Dataset]:
    train_data, valid_data, test_data = Multi30k.splits(
      exts=('.de','.en'),
      fields=(source, target),
    )

    if not Path('./.data/source.Field').is_file():
      self.build_vocab(source=source, target=target, train_data=train_data)
      self.save_fields(source=source, target=target)

    return train_data, valid_data, test_data

  def build_vocab(self, source: Field, target: Field, train_data: Dataset):
    source.build_vocab(train_data, min_freq=2)
    target.build_vocab(train_data, min_freq=2)

  def save_fields(self, source: Field, target: Field):
    with open('./.data/source.Field', 'wb') as source_file, \
      open('./.data/target.Field', 'wb') as target_file:

      dill.dump(source, source_file)
      dill.dump(target, target_file)

  def get_iterator(
    self,
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset):
    return BucketIterator.splits(
      (train_data, valid_data, test_data),
      batch_size=self.const.BATCH_SIZE,
      device=self.device)
