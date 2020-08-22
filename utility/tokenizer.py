from typing import List

import de_core_news_md
import en_core_web_md

def tokenize_de(text: str) -> List[str]:
  '''
  Tokenize German text from a string into a list of strings
  '''
  spacy_de = de_core_news_md.load()
  return [token.text for token in spacy_de.tokenizer(text)]

def tokenize_en(text: str) -> List[str]:
  '''
  Tokenize English text from a string into a list of strings
  '''
  spacy_en = en_core_web_md.load()
  return [token.text for token in spacy_en.tokenizer(text)]
  