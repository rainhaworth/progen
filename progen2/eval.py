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
MAX_ID = 29


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
    parser.add_argument('--max-window', type=int, default=-1)
    parser.add_argument('--acc', type=lambda x: (str(x).lower() == 'true'), default=False)
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
        dataset = make_gen_from_ext(args.data)

    # (4) eval

    model.eval()

    with print_time('evaluating'):
        n = 0
        prev_seq = None
        ppls = []
        i = 0
        for seq, _ in dataset:
            i += 1
            #if i < 100000: continue
            if seq == prev_seq: continue
            if len(seq) > 500 or len(seq) > 5000: continue
            else: prev_seq = seq
            print('seq:', seq)

            ce = seq_to_ce(seq, model, tokenizer, device)

            print('CE:\t', ce)
            print('PPL:\t', 2 ** ce)
            ppls.append(2**ce)

            # TODO: soft acc

            if args.acc:
                # prepare for acc computation
                seq = tokenizer.encode(seq).ids
                seq = [BOS_ID] + seq + [EOS_ID]
                seqlen = len(seq)
                seq = torch.tensor(seq).to(device)
                seq = seq[None,:]
                idxs = torch.tensor(list(range(seqlen)))
                
                #print(seq)

                # acc starting from random subseq of length 5
                w_init = 5
                w = w_init
                sub_start = np.random.randint(seqlen-w-1)
                subseq = seq[:,sub_start:sub_start+w]
                acc_r = 0.0
                for _ in tqdm(range(w, seqlen)):
                    # normal generation
                    new_token, new_pos = gen_step(model, subseq, idxs[:w], device)
                    
                    # update subseq
                    if new_pos == -1:
                        sub_start -= 1
                        tgt = seq[0, sub_start]
                        #print(seq[0, sub_start-w_init:sub_start+w_init], new_token)
                    else:
                        tgt = seq[0, sub_start + w]
                        #print(seq[0, sub_start+w-w_init:sub_start+w+w_init], new_token)
                    w += 1
                    
                    #print(i, w, sub_start, w+sub_start, new_token, tgt, new_pos)
                    subseq = seq[:,sub_start:sub_start+w]
                    #print(subseq)

                    acc_r += (new_token == tgt)#.numpy(force=True)
                
                print('hard acc r:', acc_r.numpy(force=True) / (seqlen-w_init))

                # forward only acc
                acc_f = 0.0
                for i in tqdm(range(1,seqlen-0)):
                    # get predictions
                    subseq = seq[:,:i]
                    logits, _ = gen_step(model, subseq, idxs[:i], device, return_logits=True)

                    # use raw prediction weight for next token as accuracy
                    acc_f += (torch.argmax(logits[-1]) == seq[0,i]).numpy(force=True)
                
                print('hard acc f:', acc_f / seqlen)
                
                # backward only acc
                acc_b = 0.0
                for i in tqdm(range(1,seqlen-0)):
                    # get predictions
                    subseq = seq[:,-i:]
                    logits, _ = gen_step(model, subseq, idxs[:i], device, return_logits=True)

                    # use raw prediction weight for next token as accuracy
                    acc_b += (torch.argmax(logits[0]) == seq[0,-i-1]).numpy(force=True)
                
                print('hard acc b:', acc_b / seqlen)

            print()

            n += 1
            if n > 100: break
        print('mean PPL:', np.mean(np.concatenate(ppls)))
        
        seq = '1MGHGVSRPPVVTLRPAVLDDCPVLWRWRNDPETRQASVDEREIPVDTHTRWFEETLKRFDRKLFIVSADGVDAGMVRLDIQDRDAAVSVNIAPEWRGRGVGPRALGCLSREAFGPLALLRMSAVVKRENAASRIAFERAGFTVVDTGGPLLHSSKARLHVVAAIQARMGSTRLPGKVLVSIAGRPTIQRIAERLAVCQELDAVAVSTSVENRDDAIADLAAHLGLVCVRGSETDLIERLGRTAARTGADALVRITADCPLVDPALVDRVVGVWRRSAGRLEYVSNVFPPTFPDGLDVEVLSRTVLERLDREVSDPFFRESLTAYVREHPAAFEIANVEHPEDLSRLRWTMDYPEDLAFVEAVYRRLGNQGEIFGMDDLLRLLEWSPELRDLNRCREDVTVERGIRGTGYHAALRARGQAP2'
        print('baseline:', seq)

        ce = seq_to_ce(seq, model, tokenizer, device)

        print('CE:\t', ce)
        print('PPL:\t', 2 ** ce, end='\n\n')

if __name__ == '__main__':
    main()
    print('done.')
