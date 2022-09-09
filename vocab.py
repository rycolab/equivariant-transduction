import torch

class EquivariantVocab(object):
    """
    Indexes all n vocabulary words, with a subset of words which will be 
    equivariant. These equivariant words will be indexed as the first 
    len(equivariant_words) words in the vocabulary.
    """
    def __init__(self, equivariant_words, padding=True):
        self.n = 0
        self.equivariant_words = equivariant_words
        self.word2idx = {}
        self.idx2word = {}
        self.padding = padding
        self.padding_idx = -1
        for word in equivariant_words:
            self.add_word(word)
    
    def add_word(self, word):
        if word in self.word2idx.keys():
            return
        else:
            self.word2idx[word] = self.n
            self.idx2word[self.n] = word
            self.n += 1
    
    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)
    
    def change_ordering(self):
        """
        Ensures equivariant words are in the first p positions.
        """
        non_equi_words = [word for word in self.word2idx.keys() 
                            if word not in self.equivariant_words]
        for i, word in enumerate(self.equivariant_words):
            self.word2idx[word] = i
            self.idx2word[i] = word
        for j, word in enumerate(non_equi_words):
            self.word2idx[word] = j + len(self.equivariant_words)
            self.idx2word[j + len(self.equivariant_words)] = word
        if self.padding:
            self.padding_idx = self.word2idx["<EOS>"]


    def __len__(self):
        return self.n

    def tensor_to_sent(self, ten):
        idx_arr = torch.nonzero(ten.squeeze(2))
        sent = idx_arr[:,1]
        return sent
    
    def idx_to_word(self, idx):
        return self.idx2word[idx]
    
    def sent_to_tensor(self, sent):
        ten = torch.zeros((len(sent), self.n))
        ten[[i for i in range(len(sent))], sent] = 1
        return ten.unsqueeze(2)
    
    def word_to_idx(self, word):
        return self.word2idx[word]
    
    def batch_tensor_to_sent(self, ten):
        idx_arr = torch.nonzero(ten.squeeze(2))
        sent = idx_arr[:, 2]
        sent = sent.view(ten.shape[0], ten.shape[1])
        return sent
    
    def batch_sent_to_tensor(self, sent):
        B = sent.shape[0]
        N = sent.shape[1]
        ten = torch.zeros(B, N, self.n)
        ten[[i for i in range(B) for j in range(N)],
            [i for j in range(B) for i in range(N)], sent.flatten()] = 1
        return ten.unsqueeze(-1)

    def words_to_tensor(self, sent):
        input_split = sent.split(" ")
        idxs = torch.zeros((len(input_split)), dtype=torch.int64)
        for i in range(len(input_split)):
            idxs[i] = self.word_to_idx(input_split[i])
        return self.sent_to_tensor(idxs)