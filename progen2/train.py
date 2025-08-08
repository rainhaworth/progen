# finetuning script modified from sample.py

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
from models.progen.configuration_progen import ProGenConfig
import json

# import custom dataset
from models.progen.data import ProteinBindingData

# it looks like they were already doing their own native pytorch stuff here
# so let's do native pytorch training
from transformers import get_scheduler
from tqdm.auto import tqdm


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
# main


def main():
    
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config-1.4b')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train', type=str, default='./data/uniprot_sprot.fasta')
    parser.add_argument('--eval', type=str, default='')
    parser.add_argument('--save', type=str, default='./weights')
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=1000)
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    configf = f'./{args.config}.json'

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # (3) load

    with print_time('loading parameters'):
        with open(configf, 'r') as f:
            cj = json.load(f)
        config = ProGenConfig(
            cj['vocab_size'],
            cj['n_positions'],
            cj['n_ctx'],
            cj['n_embd'],
            cj['n_layer'],
            cj['n_head'],
            resid_pdrop=cj['resid_pdrop'],
            embd_pdrop=cj['embd_pdrop'],
            attn_pdrop=cj['embd_pdrop'],
            use_cache=False,
            bos_token_id=1,
            eos_token_id=2
        )
        model = ProGenForCausalLM(config)#create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

    # load dataset(s)
    
    # helper function; keep it small and simple for now
    def make_dataloader(dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=args.bsz, shuffle=True)

    with print_time('loading up to ' + str(args.max_samples) + ' samples from ' + args.train):
        train_dataset = ProteinBindingData(args.train, tokenizer, max_dim=args.max_length, max_samples=args.max_samples)
        train_dataloader = make_dataloader(train_dataset)

        eval_dataloader = None
        if args.eval != '':
            eval_dataset = ProteinBindingData(args.eval, tokenizer)
            eval_dataloader = make_dataloader(eval_dataset)

    print('train samples found:', len(train_dataset))

    # (4) configure training

    # default settings from https://huggingface.co/docs/transformers/v4.46.2/en/training
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
            name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
            )
    
    loss_fn = torch.nn.CrossEntropyLoss()

    model.to(device)    

    # (5) train

    model.train()

    with print_time('training'):
        for epoch in range(num_epochs):
            print('epoch', epoch)
            total_loss = 0
            batches = 0
            for seqs, attns, offsets, targets in train_dataloader:
                # put everything on the GPU
                seqs = seqs.to(device)
                attns = attns.to(device)
                offsets = offsets.to(device) # TODO: remove if remains unused
                targets = targets.to(device)

                logits = model(seqs,
                                attention_mask=attns,
                                pos_offsets=offsets).logits

                # squish logits + targets, compute loss
                # TODO: retrieve original lm_head size somehow instead of doing this
                loss = loss_fn(logits.view(-1, logits.size(-1) // 2), targets.view(-1))
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # print + update loss; if you want the full loss curve over the epoch, remove `end='\r'`
                total_loss += loss.item()
                batches += 1
                print('loss: {:.5f}'.format(total_loss / batches), end='\r')
            print('loss: {:.5f}\n'.format(total_loss / batches))
            
    # (6) save weights

    save_path = os.path.join(args.save, 'model.pt')
    torch.save(model, save_path)
    print('saved to', save_path, end='\n\n')


if __name__ == '__main__':
    main()
    print('done.')
