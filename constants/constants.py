from dataclasses import dataclass
from typing import ClassVar

@dataclass
class Constants:
    LEARNING_RATE: int = 0.0005
    NUMBER_OF_EPOCHS: int = 10
    BATCH_SIZE: int = 128
    SEED: int = 1234
    CLIP: int = 1