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
