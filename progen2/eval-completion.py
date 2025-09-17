# eval script modified from sample.py

# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import numpy as np
import argparse

import torch

from tokenizers import Tokenizer
from models.progen.modeling_flexible import ProGenForCausalLM

# import custom dataset
from models.progen.data import make_gen_from_ext
from models.progen.mask import idx_to_segments

# import generation step function
from generate import gen_step

from tqdm import tqdm


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VALID_AAS = 'ACDEFGHIKLMNPQRSTVWY' # restrict generation to 20 standard amino acids


########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic



########################################################################
# model


def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


########################################################################
# sample

# from sequence + list of valid indices, generate mask for inference
def make_inference_mask(seqlen, idx, device, dim=512):
    # reduce to subsequence if necessary
    assert idx[-1] < seqlen
    assert idx[0] >= 0

    # make mask
    sz = min(seqlen, dim) # TODO: handle reaching max dim better
    mask = torch.zeros((sz,sz)).to(device)
    mask[:,idx] = 1 # unmask entire columns

    # add batch dim
    return mask[None,:,:]

# greedy sampling: find best logit and return corresponding position + token
def greedy_sample(logits):
    vals, toks = torch.max(logits, dim=-1)
    best_i = torch.argmax(vals)
    return best_i, toks[best_i]

# nucleus sampling: choose position + token with probability given by logit distribution
def nucleus_sample(logits, p=0.95):
    # find largest cutoff where we retain at least p of the probability mass
    logits_rev = logits.flatten().sort(descending=True)[0]
    cum_probs = logits_rev.cumsum(0)
    min_keep_idx = torch.sum(cum_probs < p)

    # rescale logits
    min_keep_val = logits_rev[min_keep_idx]
    p_prime = cum_probs[min_keep_idx]
    logits_rescaled = torch.where(logits >= min_keep_val, logits / p_prime, 0)

    # sample; need to flatten then convert back to dim 0, dim 1 indices
    idx_flat = torch.multinomial(logits_rescaled.flatten(), 1)[0]
    return idx_flat // logits.shape[1], idx_flat % logits.shape[1]

########################################################################
# eval

def cross_entropy_2way(seq, logits):
    half_sz = logits.size(-1) // 2
    p_logits = logits[1:,:half_sz]
    n_logits = logits[:-1,half_sz:]
    p_toks = seq[0,:-1]
    n_toks = seq[0,1:]

    ce = [torch.nn.functional.cross_entropy(p_logits, p_toks), torch.nn.functional.cross_entropy(n_logits, n_toks)]
    #ce /= 2
    ce = np.array([x.numpy(force=True) for x in ce])
    return ce

def seq_to_ce(seq, model, tokenizer, device):
    idxs = list(range(len(seq)))
    seq = tokenizer.encode(seq).ids
    seq = torch.tensor(seq).to(device)
    seq = seq[None,:]

    mask = make_inference_mask(seq.size(1), idxs, device, seq.size(1))
    logits = model(seq, attention_mask=mask).logits
    logits = torch.squeeze(logits, 0)

    ce = cross_entropy_2way(seq, logits)
    return ce


########################################################################
# main


def main():
    
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/model.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--data', type=str, default='./data/uniprot_sprot.fasta')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--rep-window', type=int, default=4)
    parser.add_argument('--rep-penalty', type=float, default=1.2)
    parser.add_argument('--sample', choices=['nucleus', 'greedy'], default='nucleus')
    parser.add_argument('--max-window', type=int, default=-1)
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    #ckpt = args.model

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # (3) load

    with print_time('loading model'):
        #model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)
        model = torch.load(args.weights, weights_only=False)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

        # get valid token IDs; does not work with proper BPE
        # this excludes terminals, which are handled later
        valid_ids = tokenizer.encode(VALID_AAS).ids
        invalid_ids = [x for x in range(32) if x not in valid_ids]

    # load dataset

    with print_time('loading datasets'):
        dataset = make_gen_from_ext(args.data)

    # (4) eval

    model.eval()

    keep_fracs = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    with print_time('evaluating'):
        n = 0
        prev_seq = None
        ppls = []
        for seq, _ in dataset:
            if seq == prev_seq: continue
            if len(seq) < 100 or len(seq) > 5000: continue
            else: prev_seq = seq
            print('seq:', seq)

            init_seq = seq

            # 2 problem settings: contiguous + fragmented subseq
            for contiguous in [False,True]:
                for keep_frac in keep_fracs:
                    # get subseq
                    keep_sz = max(1, int(keep_frac * len(seq)))
                    
                    if contiguous:
                        keep_start = np.random.randint(0, len(seq)-keep_sz+1)
                        keep_idx = np.arange(keep_start, keep_start+keep_sz)
                    else:
                        keep_idx = np.sort(np.random.choice(range(len(seq)), keep_sz, replace=False))
                    
                    # make tensors
                    seq = init_seq[keep_idx[0]:keep_idx[-1]+1]
                    seq = tokenizer.encode(seq).ids
                    seq = [BOS_ID] + seq + [EOS_ID]
                    seqlen = len(seq)
                    seq = torch.tensor(seq).to(device)
                    seq = seq[None,:]
                    idxs = torch.tensor(keep_idx - keep_idx[0])
                    
                    # generate
                    # TODO: constrain to fill in the middle if non contiguous
                    gen_steps = len(seq) - keep_sz
                    for _ in range(gen_steps):
                        # generate next token if possible
                        new_token, new_pos = gen_step(model, seq, idxs, device, invalid_ids)
                        if new_token == None: break

                        # update seq and idxs
                        new_token = new_token[None,None]
                        if new_pos == -1:
                            # prepend, shift all indices up
                            seq = torch.cat([new_token, seq], dim=-1)
                            idxs = torch.cat([new_pos[None], idxs])
                            idxs += 1
                        elif new_pos == seq.size(1):
                            # append
                            seq = torch.cat([seq, new_token], dim=-1)
                            idxs = torch.cat([idxs, new_pos[None]])
                        else:
                            # insert
                            seq[:,new_pos] = new_token
                            idxs = torch.cat([idxs, new_pos[None]]).sort()[0]
                            idxs = idxs.sort()[0]

                    print('generated {:.2f}%, contiguous = {}, CE {}: {}'.format(
                        (1-keep_frac)*100,
                        contiguous,
                        seq_to_ce(seq, model, tokenizer, device),
                        tokenizer.decode(seq.squeeze().numpy(force=True))
                        )
                    )

if __name__ == '__main__':
    main()
    print('done.')
