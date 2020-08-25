from .attention import MultiHeadAttention

import torch.nn as nn
import torch

# ref: Transformerのデータの流れを追ってみる
# https://qiita.com/FuwaraMiyasaki/items/239f3528053889847825
# no need to understand Japanese, just read the detail from image_2

# heavily borrow from:
# https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py

class TransformerBlock(nn.Module):
  def __init__(self, embedding_size, number_of_heads, forward_expansion, dropout=0.1):
    super(TransformerBlock, self).__init__()

    self.attention = MultiHeadAttention(embedding_size=embedding_size, number_of_heads=number_of_heads)
    self.norm1 = nn.LayerNorm(embedding_size)
    self.norm2 = nn.LayerNorm(embedding_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embedding_size, forward_expansion * embedding_size),
      nn.ReLU(),
      nn.Linear(forward_expansion * embedding_size, embedding_size),
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, query, value, key, mask):
    attention = self.attention(query=query, value=value, key=key, mask=mask)

    # Add skip connection, run through normalization and dropout
    x = self.dropout(self.norm1(attention + query))
    forward = self.feed_forward(x)
    out = self.dropout(self.norm2(forward + x))

    return out

class Encoder(nn.Module):
  def __init__(
    self,
    source_vocab_size,
    embedding_size,
    number_of_layers,
    number_of_heads,
    forward_expansion,
    max_length,
    dropout=0.1,
    device='cuda',
  ):

    super(Encoder, self).__init__()

    self.embedding_size = embedding_size
    self.device = device
    self.word_embedding = nn.Embedding(source_vocab_size, embedding_size)
    self.position_embedding = nn.Embedding(max_length, embedding_size)
    self.dropout = nn.Dropout(dropout)

    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          embedding_size=embedding_size,
          number_of_heads=number_of_heads,
          forward_expansion=forward_expansion,
          dropout=dropout,
        )
        for _ in range(number_of_layers)
      ]
    )

  def forward(self, x, mask):
    batch_size, source_len = x.shape
    
    positions = torch.arange(0, source_len).expand(batch_size, source_len).to(self.device)
    # Embedded source
    out = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
    # In encoder, value, query, key are all the same
    for layer in self.layers:
      out = layer(query=out, value=out, key=out, mask=mask)

    return out

class DecoderBlock(nn.Module):
  def __init__(
    self,
    embedding_size,
    number_of_heads,
    forward_expansion,
    dropout=0.1,
  ):

    super(DecoderBlock, self).__init__()

    self.norm = nn.LayerNorm(embedding_size)
    self.attention = MultiHeadAttention(embedding_size=embedding_size, number_of_heads=number_of_heads)
    self.dropout = nn.Dropout(dropout)

    self.transformer_block = TransformerBlock(
      embedding_size=embedding_size,
      number_of_heads=number_of_heads,
      forward_expansion=forward_expansion,
      dropout=0.1
    )
  
  def forward(self, target, value, key, source_mask, target_mask):
    attention = self.attention(value=target, key=target, query=target, mask=target_mask)
    query = self.dropout(self.norm(attention + target))
    out = self.transformer_block(query=query, value=value, key=key, mask=source_mask)

    return out

class Decoder(nn.Module):
  def __init__(
    self,
    target_vocab_size,
    embedding_size,
    number_of_layers,
    number_of_heads,
    forward_expansion,
    max_length,
    dropout=0.1,
    device='cuda',
  ):

    super(Decoder, self).__init__()

    self.device = device
    self.number_of_layers = number_of_layers
    self.max_length = max_length

    self.word_embedding = nn.Embedding(target_vocab_size, embedding_size)
    self.position_embedding = nn.Embedding(max_length, embedding_size)
    self.dropout = nn.Dropout(dropout)
    self.full_connection = nn.Linear(embedding_size, target_vocab_size)

    self.layers = nn.ModuleList(
      [
        DecoderBlock(
          embedding_size=embedding_size,
          number_of_heads=number_of_heads,
          forward_expansion=forward_expansion,
          dropout=dropout
        )
        for _ in range(self.number_of_layers)
      ]
    )

  def forward(self, target, encoder_output, source_mask, target_mask):
    batch_size, target_len = target.shape
    positions = torch.arange(0, target_len).expand(batch_size, target_len).to(self.device)
    target = self.dropout((self.word_embedding(target) + self.position_embedding(positions)))
    
    for layer in self.layers:
      target = layer(
        target=target,
        value=encoder_output,
        key=encoder_output,
        source_mask=source_mask,
        target_mask=target_mask
      )

    out = self.full_connection(target)

    return out

class Transformer(nn.Module):
  def __init__(
    self,
    source_vocab_size,
    target_vocab_size,
    source_padding_index,
    target_padding_index,
    embedding_size=512,
    number_of_layers=6,
    forward_expansion=4,
    number_of_heads=8,
    dropout=0.1,
    device='cuda',
    max_length=128,
  ):

    super(Transformer, self).__init__()

    self.encoder = Encoder(
      source_vocab_size=source_vocab_size,
      embedding_size=embedding_size,
      number_of_layers=number_of_layers,
      number_of_heads=number_of_heads,
      forward_expansion=forward_expansion,
      max_length=max_length,
      dropout=dropout,
      device=device,
    )

    self.decoder = Decoder(
      target_vocab_size=target_vocab_size,
      embedding_size=embedding_size,
      number_of_layers=number_of_layers,
      number_of_heads=number_of_heads,
      forward_expansion=forward_expansion,
      max_length=max_length,
      dropout=dropout,
      device=device,
    )

    self.source_padding_index = source_padding_index
    self.target_padding_index = target_padding_index
    self.device = device

  def make_source_mask(self, source):
    # [batch_size, 1, 1, source_len]
    source_mask = (source != self.source_padding_index).unsqueeze(1).unsqueeze(2)
    return source_mask.to(self.device)

  def make_target_mask(self, target):
    batch_size, target_len = target.shape
    # [batch_size, 1, 1, target_len]
    target_mask = (target != self.target_padding_index).unsqueeze(1).unsqueeze(2)
    target_sub_mask = torch.tril(torch.ones((target_len, target_len))).bool().to(self.device)
    # [batch_size, 1, target_len, target_len]
    target_mask = target_mask & target_sub_mask

    return target_mask.to(self.device)

  def forward(self, source, target):
    source_mask = self.make_source_mask(source)
    target_mask = self.make_target_mask(target)
    encoder_output = self.encoder(x=source, mask=source_mask)

    out = self.decoder(
      target=target,
      encoder_output=encoder_output,
      source_mask=source_mask,
      target_mask=target_mask,
    )

    return out

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