from model.transformer import Transformer
from constants.constants import Constants

import utility.model as model_utils

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

import math
import time

class Trainer(object):
  def __init__(
    self,
    const: Constants,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_iterator,
    valid_iterator,
    device
    ):

    self.device = device
    self.const = const
    self.data_loader = data_loader
    self.optimizer = optimizer

    self.train_iterator = train_iterator
    self.valid_iterator = valid_iterator
    
    print(f'The model has {model_utils.count_parameters(model):,} trainable parameters')

    self.criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    writer = SummaryWriter(f'runs/loss_plot')
    step = 0

  def train_step():
    self.model.train()
    
    epoch_loss = 0
    
    for i,batch in enumerate(self.train_iterator):
        
      src = batch.src
      trg = batch.trg
      
      self.optimizer.zero_grad()
      output = self.model(src, trg[:, :-1])
      output_dim = output.shape[-1]

      output = output.contiguous().view(-1, output_dim)
      trg = trg[:, 1:].contiguous().view(-1)

      loss = self.criterion(output, trg)

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), self.const.CLIP)

      optimizer.step()
      
      epoch_loss += loss.item()

      return epoch_loss / len(iterator), loss

  def evaluate_step():
    self.model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
      for i, batch in enumerate(self.valid_iterator):
        src = batch.src
        trg = batch.trg

        output = model(src,trg[:, :-1])    
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = self.criterion(output,trg)

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

  def train():
    best_valid_loss = float('inf')

    for epoch in range(self.const.NUMBER_OF_EPOCHS):

      checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
      model_utils.save_checkpoint(checkpoint)

      start_time = time.time()
      
      train_loss, loss = self.train_step()
      valid_loss = self.evaluate_step()
      
      end_time = time.time()
      
      epoch_mins, epoch_secs = model_utils.epoch_time(start_time, end_time)

      # Plot to tensorboard
      writer.add_scalar('Training loss', loss, global_step=epoch * 100)
      
      if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),'checkpoints/model.best.pt')
      
      print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

      if (epoch == 0 or epoch == self.const.NUMBER_OF_EPOCHS):
        if epoch == 0 :
          print('starting time: \n')
        else:
          print('ending time: \n')
        
        current_time = time.localtime().strftime('%H:%M:%S',t)
        print('Local Time : ',current_time)
