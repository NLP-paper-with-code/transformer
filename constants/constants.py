def constant(function):
  def _set(self, value):
      raise TypeError
  def _get(self):
      return function()
  return property(_set, _get)

class Constants(object):
  @constant
  def BATCH_SIZE() -> int:
    return 128

  @constant
  def SEED() -> int:
    return 1234

  @constant
  def LEARNING_RATE() -> int:
    return 0.0005

  @constant
  def NUMBER_OF_EPOCHS() -> int:
    return 10

  @constant
  def CLIP() -> int:
    return 1