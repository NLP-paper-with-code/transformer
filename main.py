from model.transformer import Transformer

import torch

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  source = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
  target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

  source_padding_index = 0
  target_padding_index = 0
  source_vocab_size = 10
  target_vocab_size = 10

  model = Transformer(
    source_vocab_size=source_vocab_size,
    target_vocab_size=target_vocab_size,
    source_padding_index=source_padding_index,
    target_padding_index=target_padding_index,
    device=device
  ).to(device)

  out = model(source=source, target=target[:, :-1])
  
  print(out)
  print(out.shape)  