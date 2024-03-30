from vocab import vocab


def data_process(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    char_list = list(set(content))

    return char_list

def get