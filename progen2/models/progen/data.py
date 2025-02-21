import torch
from torch.utils.data import Dataset

from .mask import idx_to_mask_start, rand_mask_start

class ProteinBindingData(Dataset):
    def __init__(self, fasta_file, tokenizer, max_dim=512, max_samples=1000, rand_bindings=False):
        self.max_dim = max_dim
        # load entire dataset into working memory
        # if you want to use a big dataset rewrite as iterable
        # or bring lots of memory i guess
        self.seqs = []
        self.attns = []
        self.offsets = []
        self.targets = []

        # for now, only support random bindings
        with open(fasta_file) as f:
            seq = ''
            sample_count = 0
            for line in f:
                if len(line) == 0 or line[0] == '>':
                    if len(seq) == 0:
                        continue
                    # for now, trim
                    seq = seq[:max_dim-2]
                    # tokenize
                    seq = tokenizer.encode(seq).ids
                    # add BOS, EOS; see tokenizer.json
                    seq = [1] + seq + [2]
                    # store
                    self.seqs.append(torch.tensor(seq))
                    self.attns.append(None)
                    self.offsets.append(None)
                    self.targets.append(None)

                    # have we hit max_samples?
                    sample_count += 1
                    if sample_count >= max_samples:
                        break
                    
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
            targets = self.targets[idx]
        else:
            # for now, pad everything to max_dim
            attn, offset, targets = rand_mask_start(len(seq), self.max_dim)

        # pad sequence
        if len(seq) < self.max_dim:
            seq = torch.cat(( seq, torch.zeros(self.max_dim - len(seq)) )).to(int)

        # convert targets from indices to token ids
        targets = torch.tensor(targets)
        targets = torch.where(targets >= 0, seq[targets], targets)

        return seq, attn, offset, targets

# generation dataset
class ProteinBindingOnlyData(Dataset):
    def __init__(self, fasta_file, tokenizer, max_dim=512, max_samples=15, rand_bindings=False):
        self.max_dim = max_dim
        self.seqs = []
        self.idxs = []

        # for now, only support random bindings
        with open(fasta_file) as f:
            seq = ''
            sample_count = 0
            for line in f:
                if len(line) == 0 or line[0] == '>':
                    if len(seq) == 0:
                        continue
                    # for now, trim
                    seq = seq[:max_dim-2]
                    # tokenize
                    seq = tokenizer.encode(seq).ids
                    # add BOS, EOS; see tokenizer.json
                    seq = [1] + seq + [2]
                    # extract random binding site
                    bind_idx = rand_mask_start(len(seq), self.max_dim, just_binding=True)
                    seq = seq[bind_idx[0] : bind_idx[-1] + 1]
                    # store
                    self.seqs.append(torch.tensor(seq))
                    self.idxs.append(bind_idx - bind_idx[0])

                    # have we hit max_samples?
                    sample_count += 1
                    if sample_count >= max_samples:
                        break
                    
                    # reset
                    seq = ''
                seq += line.strip()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        idxs = self.idxs[idx]

        return seq, idxs
