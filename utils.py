import torch
from torch.nn.utils.rnn import pad_sequence
import random
from vocab import vocab


def get_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    char_list = list(set(content))

    return char_list


def data_process(file_path):
    poems = []  # 存储每首诗的列表
    current_poem = ""  # 当前正在录入的诗的字符串

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 遍历每行内容
    for line in lines:
        line = line.strip()  # 去除行尾的换行符和空格

        # 如果行不为空，则添加到当前诗的字符串中
        if line != "":
            current_poem += line
        # 如果行为空，且当前诗的字符串不为空，则将当前诗添加到诗列表中，并重置当前诗的字符串
        elif current_poem != "":
            poems.append(current_poem)
            current_poem = ""

    # 如果最后一个诗的字符串不为空，则将其添加到诗列表中
    if current_poem != "":
        poems.append(current_poem)

    data_set = [torch.tensor(vocab.sentence2ids((list(poem).append("<end>")))) for poem in poems]

    return data_set


def make_batch(data_set, batch_size):
    batch = random.sample(data_set, k=batch_size)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=vocab.vocab_size - 1)
    return padded_batch
