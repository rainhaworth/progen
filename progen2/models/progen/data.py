import torch
from torch.utils.data import Dataset

from .mask import idx_to_mask_start, rand_mask_start

class ProteinBindingData(Dataset):
    def __init__(self, fasta_file, tokenizer, rand_bindings=False):
        # load entire dataset into working memory
        # if you want to use a big dataset rewrite as iterable
        # or bring lots of memory i guess
        self.seqs = []
        self.attns = []
        self.offsets = []

        # for now, only support random bindings
        with open(fasta_file) as f:
            for line in f:
                if len(line) == 0 or line[0] == '>':
                    continue
                self.seqs.append(tokenizer.encode(line).ids)
                self.attns.append(None)
                self.offsets.append(None)

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
