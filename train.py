from model.transformer import Transformer
from constants.constants import Constants

import utility.model as model_utils

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

import torch.nn as nn
import torch

from typing import Tuple

import math
import time

class Trainer(object):
  def __init__(
    self,
    const: Constants,
    optimizer: torch.optim.Optimizer,
    criterion,
    device,
    ):

    self.optimizer = optimizer
    self.criterion = criterion
    self.const = const

    self.device = device

  def train_step(
    self,
    epoch: int,
    model: nn.Module,
    train_iterator) -> Tuple[float]:
    
    model.train()
    epoch_loss = 0

    steps = tqdm(enumerate(train_iterator), total=len(train_iterator), leave=False)

    for step, batch in steps:
      
      src = batch.src.to(self.device)
      trg = batch.trg.to(self.device)
      
      self.optimizer.zero_grad()

      output = model(src, trg[:, :-1])
      output_dim = output.shape[-1]

      output = output.contiguous().view(-1, output_dim)
      trg = trg[:, 1:].contiguous().view(-1)

      loss = self.criterion(output, trg)

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), self.const.CLIP)

      self.optimizer.step()
      
      epoch_loss += loss.item()

      steps.set_description(f'Epoch [{epoch}/{self.const.NUMBER_OF_EPOCHS}][train]')
      steps.set_postfix(loss=epoch_loss / len(train_iterator))

    return epoch_loss / len(train_iterator), loss

  def evaluate_step(
    self,
    epoch: int,
    model: nn.Module,
    evaluate_iterator) -> float:

    model.eval()
    epoch_loss = 0
    steps = len(evaluate_iterator)

    steps = tqdm(enumerate(train_iterator), total=len(train_iterator), leave=False)

    with torch.no_grad():
      for step, batch in steps:
        src = batch.src.to(self.device)
        trg = batch.trg.to(self.device)

        output = model(src, trg[:, :-1])    
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = self.criterion(output, trg)

        epoch_loss += loss.item()

        steps.set_description(f'[{epoch}/{self.const.NUMBER_OF_EPOCHS}][evaluate]')
        steps.set_postfix(loss=epoch_loss / len(evaluate_iterator))

    return epoch_loss / len(evaluate_iterator)

  def train(
    self,
    model: nn.Module,
    train_iterator,
    valid_iterator):
    writer = SummaryWriter(f'runs/loss_plot')
    best_valid_loss = float('inf')

    for epoch in range(self.const.NUMBER_OF_EPOCHS):
      checkpoint = {'state_dict': model.state_dict(),'optimizer': self.optimizer.state_dict()}
      model_utils.save_checkpoint(checkpoint)

      start_time = time.time()

      train_loss, loss = self.train_step(model=model, epoch=epoch, train_iterator=train_iterator)
      valid_loss = self.evaluate_step(model=model, epoch=epoch, evaluate_iterator=valid_iterator)

      end_time = time.time()
      
      epoch_mins, epoch_secs = model_utils.epoch_time(start_time, end_time)

      # Plot to tensorboard
      writer.add_scalar('Training loss', loss, global_step=epoch * 100)
      
      if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),'./checkpoints/model.best.pt')

      print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')

      if (epoch == 0 or epoch == self.const.NUMBER_OF_EPOCHS):
        if epoch == 0 :
          print('starting time: \n')
        else:
          print('ending time: \n')
        
        current_time = time.strftime('%H:%M:%S', time.localtime())
        print('Local Time : ', current_time)

  def test(self, model: nn.Module, test_iterator):
    test_loss = self.evaluate_step(model=model, evaluate_iterator=test_iterator)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')