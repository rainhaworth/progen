import torch
from torch.utils.data import Dataset
import csv
import numpy as np

from .mask import idx_to_mask_start, rand_mask_start

# FASTA reader
def fasta_gen(file):
    with open(file) as f:
        seq = ''
        for line in f:
            if len(line) == 0 or line[0] == '>':
                if len(seq) == 0:
                    continue
                yield seq
                seq = ''
            seq += line.strip()

# TSV reader (for UniProt ID mapper output w/ binding sites)
def tsv_gen(file):
    with open(file) as f:
        reader = csv.reader(f, delimiter='\t')
        col_idxs = None
        col_names = ['Sequence', 'Binding site']
        for row in reader:
            # find columns that contain the data we're interested in
            if col_idxs is None:
                col_idxs = [row.index(col_names[0]), row.index(col_names[1])]
                continue
            # grab sequence and raw binding site data string
            seq = row[col_idxs[0]]
            bind = row[col_idxs[1]]
            # parse binding site data
            bind_split = bind.split(';')
            for sub_bind in bind_split:
                # for now, just make pairs for each BINDING instance
                if sub_bind[:7] == 'BINDING':
                    # get single position index or range of position indices
                    bind_range = sub_bind.split()[-1].split('..')
                    assert 1 <= len(bind_range) <= 2
                    if len(bind_range) == 1:
                        # single -> tensor
                        bind_idx = torch.tensor(int(bind_range[0]))
                    else
                        # range
                        bind_idx = range(int(bind_range[0]), int(bind_range[1])+1)
                        bind_idx = torch.tensor(bind_idx)
                    yield seq, bind_idx

# binding site dropout for tensor idxs
def apply_dropout(idxs, p_drop=0.2):
    if len(idxs) <= 1:
        return idxs
    elems_to_drop = np.random.binomial(len(idxs), p_drop)
    elems_to_keep = max(len(idxs) - elems_to_drop, 1)
    idxs_new = idxs[torch.randperm(len(idxs))]
    return torch.sort(idxs_new[:elems_to_keep])

class ProteinBindingData(Dataset):
    def __init__(self, in_file, tokenizer, max_dim=512, max_samples=1000, rand_bindings=True, p_drop=0.2):
        self.max_dim = max_dim
        self.p_drop = p_drop
        # load entire dataset into working memory
        # if you want to use a big dataset rewrite as iterable
        # or bring lots of memory i guess
        self.seqs = []
        self.idxs = []

        # set generator type
        if rand_bindings:
            gen = fasta_gen(in_file)
        else:
            gen = tsv_gen(in_file)

        # fetch all sequences and binding sites if available
        sample_count = 0
        for seq, idx in gen:
            # for now, trim
            seq = seq[:max_dim-2]
            # tokenize
            seq = tokenizer.encode(seq).ids
            # add BOS, EOS; see tokenizer.json
            seq = [1] + seq + [2]
            # store
            self.seqs.append(torch.tensor(seq))
            self.idxs.append(idx)

            # have we hit max_samples?
            sample_count += 1
            if sample_count >= max_samples:
                break

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        idxs = self.idxs[idx]

        # generate random path -> mask + targets
        # for now, pad everything to max_dim
        if idxs is not None:
            # apply dropout
            idxs_drop = apply_dropout(idxs, self.p_drop)
            # generate
            attn, offset, targets = idx_to_mask_start(idxs_drop, len(seq), self.max_dim)
        else:
            attn, offset, targets = rand_mask_start(len(seq), self.max_dim, p_drop=self.p_drop)

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

        # set generator type
        if rand_bindings:
            gen = fasta_gen(in_file)
        else:
            gen = tsv_gen(in_file)
        
        # fetch all
        for seq, idx in gen:
            # for now, trim
            seq = seq[:max_dim-2]
            # tokenize
            seq = tokenizer.encode(seq).ids
            # add BOS, EOS; see tokenizer.json
            seq = [1] + seq + [2]
            # get binding site if necessary
            if idx is None:
                idx = rand_mask_start(len(seq), self.max_dim, just_binding=True)
            seq = seq[idx[0] : idx[-1] + 1]
            # store
            self.seqs.append(torch.tensor(seq))
            self.idxs.append(idx - idx[0])

            # have we hit max_samples?
            sample_count += 1
            if sample_count >= max_samples:
                break

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        idxs = self.idxs[idx]

        return seq, idxs
