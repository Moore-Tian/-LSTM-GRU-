import torch
import torch.nn
from modle import *
from utils import *
from vocab import *


file_path = "dataset/poetryFromTang.txt"

char_list = get_dataset(file_path)
print(char_list)

vocab = Vocab(char_list)