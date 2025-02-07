import torch
from torch.utils.data import Dataset

from .mask import idx_to_mask_start, rand_mask_start

class ProteinBindingData(Dataset):
    def __init__(self, fasta_file, tokenizer, max_dim=512, rand_bindings=False):
        # load entire dataset into working memory
        # if you want to use a big dataset rewrite as iterable
        # or bring lots of memory i guess
        self.seqs = []
        self.attns = []
        self.offsets = []

        # for now, only support random bindings
        with open(fasta_file) as f:
            seq = ''
            it = 0
            for line in f:
                it += 1
                if it > 1000:
                    break
                if len(line) == 0 or line[0] == '>':
                    if seq == '':
                        continue
                    seq = seq[:max_dim]
                    # tokenize
                    seq = tokenizer.encode(seq).ids
                    # pad
                    if len(seq) < max_dim:
                        seq = seq + [0] * (max_dim - len(seq))
                    # write
                    self.seqs.append(torch.tensor(seq))
                    self.attns.append(None)
                    self.offsets.append(None)
                    
                    # reset
                    seq = ''
                seq += line.strip()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        seq = self.seqs[idx]

        # return attn and mask if known, otherwise generate
        if self.attns[idx] is not None:
            attn = self.attns[idx]
            offset = self.offsets[idx]
        else:
            attn, offset = rand_mask_start(len(seq))

        return seq, attn, offset
