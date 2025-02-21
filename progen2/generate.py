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


########################################################################
# main


def main():
    
    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B
    
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-small')
    parser.add_argument('--weights', type=str, default='./weights/model.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--data', type=str, default='./data/uniprot_sprot.fasta')
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    ckpt = args.model

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

    # TODO: write all these things
        # binding-site-only Dataset w/ appropriate masking
        # sampling algorithm (greedy for now)
        # procedure to step through sampling until we hit EOS + BOS

    BOS_ID = 1
    EOS_ID = 2

    max_steps = 10

    model.eval()

    with print_time('generating'):
        for seq, idxs in dataloader:
            print('binding site:\t', tokenizer.decode(seq.squeeze().numpy()))
            print('idxs:\t\t', idxs.squeeze().numpy().tolist())
            # put everything on the GPU
            seq = seq.to(device)
            idxs = idxs.to(device)

            idxs = torch.squeeze(idxs)

            # greedy search
            for _ in range(max_steps):
                # make mask, call model, squeeze batch dim to make life easier
                mask = make_inference_mask(seq.size(1), idxs, device)
                logits = model(seq, attention_mask=mask).logits
                logits = torch.squeeze(logits, 0)

                # get segments; use copy of idxs so we don't have weird memory issues
                segments = idx_to_segments(idxs.detach().clone())

                # get indices for PTP and NTP
                p_idxs = [seg[0] for seg in segments]
                n_idxs = [seg[1] for seg in segments]

                # TODO: filter out PTP at BOS, NTP at EOS

                # find PTP and NTP logits at corresponding indices
                half_sz = logits.size(-1) // 2
                p_logits = logits[p_idxs,:half_sz]
                n_logits = logits[n_idxs,half_sz:]

                # get best scores at each position
                p_vals, p_toks = torch.max(p_logits, dim=-1)
                n_vals, n_toks = torch.max(n_logits, dim=-1)

                # get single best score for PTP and NTP
                bpv, bpi = torch.max(p_vals, dim=0)
                bnv, bni = torch.max(n_vals, dim=0)

                # if previous better than next, do PTP
                if bpv > bnv:
                    # get new token label + position where we predicted it
                    new_token = p_toks[bpi]
                    new_pos = p_idxs[bpi]
                    # shift to previous position for insertion
                    new_pos -= 1
                # otherwise, do NTP
                else:
                    new_token = n_toks[bni]
                    new_pos = n_idxs[bni]
                    new_pos += 1

                # finally, update seq and idxs
                new_token = new_token[None,None]

                # edge cases: increase sequence length
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

                # print
                #print(idxs.numpy(force=True))
                #print(tokenizer.decode(seq.squeeze().numpy(force=True)))
            print('generated:\t', tokenizer.decode(seq.squeeze().numpy(force=True)), end='\n\n')


if __name__ == '__main__':
    main()
    print('done.')
