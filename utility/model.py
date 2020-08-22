import torch.nn as nn
import torch

def count_parameters(model: nn.Module) -> int:
  return sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad)

def initialize_weights(model: nn.Module):
  if hasattr(model, 'weight') and model.weight.dim() > 1:
    nn.init.xavier_uniform_(model.weight.data)

def load_checkpoint(
  checkpoint,
  model: nn.Module,
  optimizer: torch.optim.Optimizer):
  print('=> Loading checkpoint')
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])

def save_checkpoint(state, filename='checkpoints/my_checkpoint.pth.tar'):
  print('=> Saving checkpoint')
  torch.save(state, filename)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  
  return elapsed_mins, elapsed_secs