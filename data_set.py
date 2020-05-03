from enum import Enum

# used in dataloader and dataset to specify which specific dataset should be used
class DataSet(Enum):
    TRAIN = "train",
    VAL = "validation",
    TEST = "test"