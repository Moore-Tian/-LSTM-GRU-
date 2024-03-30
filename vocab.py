class vocab():
    def __init__(self, data_set):
        self.word2id = {}
        self.id2word = {}
        for word in data_set:
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)
                self.id2word[self.word2id[word]] = word

    def __len__(self):
        return len(self.word2id)
    
    def sentence2ids(self, sentence):
        char_list = list(sentence)
        return [self.word2id[word] for word in char_list]

    def ids2sentence(self, ids):
        return ''.join([self.id2word[id] for id in ids])
