# generation script modified from sample.py

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
from models.progen.data import ProteinBindingOnlyData
from models.progen.mask import idx_to_segments


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
# main


def main():
    
    # (0) constants

    #models_151M = [ 'progen2-small' ]
    #models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    #models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    #models_6B = [ 'progen2-xlarge' ]
    #models = models_151M + models_754M + models_2B + models_6B
    
    # (1) params

    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str, choices=models, default='progen2-small')
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

    # load dataset

    with print_time('loading datasets'):
        dataset = ProteinBindingOnlyData(args.data, tokenizer, max_samples=15)
        dataloader = torch.utils.data.DataLoader(dataset)

    # (4) generate

    BOS_ID = 1
    EOS_ID = 2

    max_steps = args.max_steps
    rw = args.rep_window
    rp = args.rep_penalty

    # sample_fn input: logits, output: (index along logits dim=0, token ID)
    if args.sample == 'nucleus':
        sample_fn = nucleus_sample
    else:
        sample_fn = greedy_sample

    model.eval()

    with print_time('generating'):
        for seq, idxs in dataloader:
            print('binding site:\t', tokenizer.decode(seq.squeeze(0).numpy()))
            print('idxs:\t\t', idxs.squeeze().numpy().tolist())
            # put everything on the GPU
            seq = seq.to(device)
            idxs = idxs.to(device)

            idxs = idxs.squeeze(0)

            # greedy search
            for _ in range(max_steps):
                # make mask, call model, squeeze batch dim to make life easier
                mask = make_inference_mask(seq.size(1), idxs, device)
                logits = model(seq, attention_mask=mask).logits
                logits = torch.squeeze(logits, 0)

                # get segments; use copy of idxs so we don't have weird memory issues
                segments = idx_to_segments(idxs.detach().clone())

                # get indices for PTP and NTP
                p_idxs = [seg[0] for seg in segments if seq[:,seg[0]] != BOS_ID]
                n_idxs = [seg[1] for seg in segments if seq[:,seg[0]] != EOS_ID]

                # find PTP and NTP logits at corresponding indices, concat
                half_sz = logits.size(-1) // 2
                if len(n_idxs) > 0 and len(p_idxs) > 0:
                    p_logits = logits[p_idxs,:half_sz]
                    n_logits = logits[n_idxs,half_sz:]
                    logits = torch.concat([p_logits, n_logits])
                
                # handle edge cases + stop if we have no valid steps
                elif len(p_idxs) > 0: logits = logits[p_idxs,:half_sz]
                elif len(n_idxs) > 0: logits = logits[n_idxs,half_sz:]
                else: break                

                # apply repetition penalties; can probably do this faster but with window=4 it's fine
                p_penalties = [[seq[:,i] for i in range(p_i, p_i+rw) if i in idxs] for p_i in p_idxs]
                n_penalties = [[seq[:,i] for i in range(n_i-rw+1, n_i+1) if i in idxs] for n_i in n_idxs]
                penalties = torch.ones_like(logits)
                for i, pens in enumerate(p_penalties + n_penalties):
                    for p in pens:
                        penalties[i, p] = rp
                logits = logits / penalties

                # compute (numerically stable) softmax over all logits representing viable next steps
                exp_logits = torch.exp(logits - torch.max(logits))
                sum_exp_logits = torch.sum(exp_logits)
                logits = exp_logits / sum_exp_logits

                # sample next step (index, token) from logits
                new_i, new_token = sample_fn(logits)

                # TODO: handle incorrect EOS/BOS output
                if new_token == BOS_ID or new_token == EOS_ID:
                    print('EOS/BOS found')

                # get new token position from index, offset according to PTP vs NTP
                PTP = new_i < len(p_idxs)
                if PTP: new_pos = p_idxs[new_i] - 1
                else:   new_pos = n_idxs[new_i - len(p_idxs)] + 1

                # finally, update seq and idxs
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

                # debugging
                #print(idxs.numpy(force=True), seq.shape)
                #print(tokenizer.decode(seq.squeeze().numpy(force=True)))
            print('generated:\t', tokenizer.decode(seq.squeeze().numpy(force=True)), end='\n\n')


if __name__ == '__main__':
    main()
    print('done.')
