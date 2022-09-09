import os
import random
import torch
from vocab import EquivariantVocab

class DataSet(object):
    def __init__(self, dir_path, in_equivariances, out_equivariances, 
        dev_percent=0.1, scan_split=None):
        self.in_equivariances = in_equivariances
        self.out_equivariances = out_equivariances
        self.dir = dir_path
        self.scan_split = scan_split
        self.data = {}
        self.dev_percent = dev_percent
        self.in_vocab = EquivariantVocab(in_equivariances)
        self.out_vocab = EquivariantVocab(out_equivariances)
        train_in, train_out = self.get_scan_data('train')
        test_in, test_out = self.get_scan_data('test')
        self.prep_vocab(self.in_vocab, train_in)
        self.prep_vocab(self.out_vocab, train_out)
        self.create_data_dict(train_in, train_out, test_in, test_out)
        self.idx = 0

    def get_scan_data(self, train_test):
        folder = {
                'simple':'simple_split', 
                'add_jump':'add_prim_split', 
                'length_generalization':'length_split', 
                'around_right':'template_split'
                }
        file_names = {
                'simple':'simple', 
                'add_jump':'addprim_jump', 
                'length_generalization':'length', 
                'around_right':'template_around_right'
                }

        if 'simple_' not in self.scan_split:
            data_folder = os.path.join(self.dir, folder[self.scan_split])
            path = os.path.join(data_folder, 
                'tasks_' + train_test + '_' + file_names[self.scan_split] 
                + '.txt')
        else:
            data_folder = os.path.join(self.dir, folder['simple'], 
                'size_variations')
            path = os.path.join(data_folder, 
                'tasks_' + train_test + '_' + self.scan_split + '.txt')
        return self.read_scan_file(path)

    def read_scan_file(self, path):
        all_lines = [l.strip("\n") for l in open(path, 'r').readlines()]
        pairs = [l.split("IN: ")[-1].split(" OUT: ") for l in all_lines]
        all_in = []
        all_out = []
        for i, p in enumerate(pairs):
            all_in.append( "<BOS> " + p[0] + " <EOS>")
            all_out.append("<BOS> " + p[1] + " <EOS>")
        return all_in, all_out

    def get_batch(self, batch_size, split='train'):
        if self.idx + batch_size > len(self.data[split]['in']):
            self.idx = 0
            if split == 'train':
                self.shuffle_train_set()
        batch_list_in = self.data[split]['in'][self.idx:self.idx+batch_size]
        batch_list_out = self.data[split]['out'][self.idx:self.idx+batch_size]
        max_length_in = max([len(t) for t in batch_list_in])
        max_length_out = max([len(t) for t in batch_list_out])
        padded_in_tensors = []
        padded_out_tensors = []
        padding_in = self.in_vocab.padding_idx
        padding_out = self.out_vocab.padding_idx
        for t, o in zip(batch_list_in, batch_list_out):
            if len(t) < max_length_in:
                pad = torch.zeros(max_length_in - t.shape[0], 
                    len(self.in_vocab), 1)
                pad[:, padding_in, :] = 1.0
                padded_in_tensors.append(torch.cat([t, pad], dim=0))
            else:
                padded_in_tensors.append(t)
            if len(o) < max_length_out:

                pad = torch.zeros(max_length_out - o.shape[0], 
                    len(self.out_vocab), 1)
                pad[:, padding_out, :] = 1.0
                padded_out_tensors.append(torch.cat([o, pad], dim=0))
            else:
                padded_out_tensors.append(o)
        self.idx += batch_size
        if self.idx > len(self.data[split]['in']):
            self.idx = 0
            if split == 'train':
                self.shuffle_train_set()
        return torch.stack(padded_in_tensors), torch.stack(padded_out_tensors)

    def shuffle_train_set(self):
        pairs = [
            (self.data['train']['in'][i], 
            self.data['train']['out'][i]) 
            for i in range(len(self.data['train']['in']))]
        random.shuffle(pairs)
        self.data['train']['in'] = [p[0] for p in pairs]
        self.data['train']['out'] = [p[1] for p in pairs]

    def prep_vocab(self, vocab, examples):
        for s in examples:
            vocab.add_sentence(s)
        vocab.change_ordering()

    def create_data_dict(self, train_in, train_out, test_in, test_out):
        self.data['train'] = {}
        self.data['test'] = {}
        self.data['dev'] = {}
        dev_cutoff = int(len(train_in) * self.dev_percent)
        if self.scan_split == 'length_generalization':
            length_sorted = sorted(zip(train_in, train_out), 
                key=lambda x:len(x[0]), reverse=True)
            train_in = [x[0] for x in length_sorted]
            train_out = [x[1] for x in length_sorted]
        self.data['train']['in'] = self.make_tensors(train_in[dev_cutoff:], 
            self.in_vocab)
        self.data['train']['out'] = self.make_tensors(train_out[dev_cutoff:], 
            self.out_vocab)
        self.data['dev']['in'] = self.make_tensors(train_in[:dev_cutoff], 
            self.in_vocab)
        self.data['dev']['out'] = self.make_tensors(train_out[:dev_cutoff], 
            self.out_vocab)
        self.data['test']['in'] = self.make_tensors(test_in, self.in_vocab)
        self.data['test']['out'] = self.make_tensors(test_out, self.out_vocab)

    def make_tensors(self, sentences, vocab):
        tensors = []
        for s in sentences:
            tensors.append(vocab.words_to_tensor(s))
        return tensors