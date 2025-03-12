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
                yield seq, None
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
                        bind_idx = torch.tensor([int(bind_range[0])])
                    else:
                        # range
                        bind_idx = range(int(bind_range[0]), int(bind_range[1])+1)
                        bind_idx = torch.tensor(bind_idx)
                    yield seq, bind_idx

# select generator from file extension
def make_gen_from_ext(file):
    ext = file.split('.')[-1]
    if ext in ['fasta', 'fa']:
        return fasta_gen(file)
    elif ext == 'tsv':
        return tsv_gen(file)
    else:
        raise ValueError('Invalid file extension ' + ext + '; expected fasta or tsv')

# binding site dropout for tensor idxs
def apply_dropout(idxs, p_drop=0.2):
    if len(idxs) <= 1:
        return idxs
    elems_to_drop = np.random.binomial(len(idxs), p_drop)
    elems_to_keep = max(len(idxs) - elems_to_drop, 1)
    idxs_new = idxs[torch.randperm(len(idxs))]
    return torch.sort(idxs_new[:elems_to_keep]).values

class ProteinBindingData(Dataset):
    def __init__(self, file, tokenizer, max_dim=512, max_samples=1000, p_drop=0.2):
        self.max_dim = max_dim
        self.p_drop = p_drop
        # load entire dataset into working memory
        # if you want to use a big dataset rewrite as iterable
        self.seqs = []
        self.idxs = []

        # get generator
        gen = make_gen_from_ext(file)

        # fetch all sequences and binding sites if available
        sample_count = 0
        for seq, idx in gen:
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

        # if sequence is bigger than max_dim, take random subsequence
        offset = 0
        if len(seq) > self.max_dim:
            min_idx = 0
            max_idx = len(seq) - self.max_dim
            # restrict to contain binding site if known
            if idxs is not None:
                # hopefully binding site is small enough to fit in the subseq
                if idxs[-1] - idxs[0] < self.max_dim:
                    min_idx = max(0, idxs[-1].item() - self.max_dim)
                    max_idx = min(max_idx, idxs[0].item())
                # if not, just get a chunk of it
                else:
                    min_idx = idxs[0].item()
                    max_idx = idxs[-1].item() - self.max_dim
                    # avoid breaking randint
                    if min_idx == max_idx:
                        max_idx += 1
            # compute offset
            offset = np.random.randint(min_idx, max_idx)
            # update seq
            seq = seq[offset : offset + self.max_dim]

        # generate random path -> mask + targets
        # for now, pad everything to max_dim
        if idxs is not None:
            # apply offset (if applicable) and dropout
            idxs -= offset
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
    def __init__(self, file, tokenizer, max_dim=512, max_samples=15):
        self.max_dim = max_dim
        self.seqs = []
        self.idxs = []

        # set generator type
        gen = make_gen_from_ext(file)
        
        # fetch all
        sample_count = 0
        for seq, idx in gen:
            # tokenize
            seq = tokenizer.encode(seq).ids
            # add BOS, EOS (see tokenizer.json)
            seq = [1] + seq + [2]
            # generate artificial binding site if necessary
            if idx is None:
                idx = rand_mask_start(len(seq), self.max_dim, just_binding=True)
            # otherwise, adjust for extra token then randomly drop indices
            else:
                idx += 1
                idx = apply_dropout(idx)
            # store smallest possible subsequence
            seq = seq[idx[0] : idx[-1] + 1]
            # trim, just in case; TODO: remove this when we can handle long seqs
            seq = seq[:max_dim]
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
