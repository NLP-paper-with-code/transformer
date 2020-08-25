from data_loader.data_loader import DataLoader
from model.transformer import Transformer
from constants.constants import Constants

import utility.inference as inference_utils
import utility.model as model_utils

import torch

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  input_sentence = 'eine frau mit einer großen geldbörse geht an einem tor vorbei .'

  const = Constants()
  data_loader = DataLoader(device=device, const=const)

  source, target = data_loader.get_fields()
  train_data, valid_data, test_data = data_loader.split_data(source=source, target=target)
  
  SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE = data_loader.get_vocab_size(
    source=source,
    target=target,
  )

  SRC_PAD_IDX, TRG_PAD_IDX = data_loader.get_padding_index(
    source=source,
    target=target,
  )

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

  model.load_state_dict(torch.load('./checkpoints/model.best.pt'))

  bleu_score = inference_utils.calculate_bleu(
    data=test_data,
    source_field=source,
    target_field=target,
    model=model,
    device=device,
  )

  print(f'BLEU score = {bleu_score*100:.2f}')