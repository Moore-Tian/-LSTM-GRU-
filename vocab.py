from utils import get_dataset


class Vocab():
    def __init__(self, data_set):
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        for word in data_set:
            if word not in self.word2id:
                self.word2id[word] = self.vocab_size
                self.id2word[self.vocab_size] = word
                self.vocab_size += 1
        self.word2id['<end>'] = self.vocab_size
        self.id2word[self.vocab_size + 1] = '<end>'
        self.vocab_size += 1

    def __len__(self):
        return self.vocab_size
    
    def sentence2ids(self, sentence):
        return [self.word2id[word] for word in sentence]

    def ids2sentence(self, ids):
        return ''.join([self.id2word[id] for id in ids])