from dataclasses import dataclass
from typing import ClassVar

@dataclass
class Constants:
    LEARNING_RATE: int = 0.0005
    NUMBER_OF_EPOCHS: int = 10
    FORWARD_EXPANSION: int = 2
    NUMBER_OF_LAYERS: int = 3
    EMBEDDING_SIZE: int = 256
    NUMBER_OF_HEADS: int = 8
    BATCH_SIZE: int = 128
    SEED: int = 1234
    CLIP: int = 1