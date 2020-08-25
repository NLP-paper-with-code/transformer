from torchtext.data.metrics import bleu_score
from torchtext.data import Field

import torchtext.data.metrics
import de_core_news_md
import torch
import spacy

import torch.nn as nn

nlp = de_core_news_md.load()

def translate_sentence(
  sentence,
  source_field: Field,
  target_field: Field,
  model: nn.Module,
  device: str,
  max_len=50
) -> str:

  model.eval()

  if isinstance(sentence, str):
    tokens = [token.text.lower() for token in nlp(sentence)]
  else:
    tokens = [token.lower() for token in sentence]

  tokens = [source_field.init_token] + tokens + [target_field.eos_token]
  
  source_indexs = [source_field.vocab.stoi[token] for token in tokens]
  source_tensor = torch.LongTensor(source_indexs).unsqueeze(0).to(device)
  source_mask = model.make_source_mask(source_tensor)

  with torch.no_grad():
    encoder_output = model.encoder(x=source_tensor, mask=source_mask)

  target_indexs = [target_field.vocab.stoi[target_field.init_token]]

  for index in range(max_len):
    target_tensor = torch.LongTensor(target_indexs).unsqueeze(0).to(device)
    target_mask = model.make_target_mask(target_tensor)

    with torch.no_grad():
      output = model.decoder(
        target=target_tensor,
        encoder_output=encoder_output,
        source_mask=source_mask,
        target_mask=target_mask,
      )

      predictedicted_token_index = output.argmax(2)[:,-1].item()
      target_indexs.append(predictedicted_token_index)

      if predictedicted_token_index == target_field.vocab.stoi[target_field.eos_token]:
        break
  
  target_tokens = [target_field.vocab.itos[index] for index in target_indexs]

  return target_tokens[1:]

def calculate_bleu(
  data,
  source_field: Field,
  target_field: Field,
  model: nn.Module,
  device: str,
  max_len=50) -> float:
    
  targets = []
  predicted_targets = []
  
  for datum in data:
    src = vars(datum)['src']
    trg = vars(datum)['trg']
    
    predicted_target = translate_sentence(
      sentence=src,
      source_field=source_field,
      target_field=target_field,
      model=model,
      device=device,
      max_len=max_len,
    )
    
    #cut off <eos> token
    predicted_target = predicted_target[:-1]
    
    predicted_targets.append(predicted_target)
    
    targets.append([trg])
      
  return bleu_score(predicted_targets, targets)
