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
# parallelism imports
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tokenizers import Tokenizer
from models.progen.modeling_flexible import ProGenForCausalLM

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
# trainer

# modified from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        loss_fn,
        gpu_id: int,
        save_every: int,
        save_dir
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.save_every = save_every
        self.save_dir = save_dir
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    def _run_batch(self, seqs, attns, targets):
        self.optimizer.zero_grad()

        logits = self.model(seqs,
                            attention_mask=attns).logits
        loss = self.loss_fn(logits.view(-1, logits.size(-1) // 2), targets.view(-1))

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        time_load = 0
        time_run = 0

        _t1 = time.time()
        for seqs, attns, _, targets in self.train_data:
            # put everything on the GPU
            seqs = seqs.to(self.gpu_id)
            attns = attns.to(self.gpu_id)
            #offsets = offsets.to(self.gpu_id) # TODO: remove if remains unused
            targets = targets.to(self.gpu_id)

            _t2 = time.time()
            loss = self._run_batch(seqs, attns, targets)

            time_load += _t2 - _t1
            time_run += time.time() - _t2

            _t1 = time.time()

            # continuous loss output; only visible in interactive mode
            print(' loss: {:.5f}'.format(loss.item()), end='\r')

        # rough loss tracking across epochs
        if self.gpu_id == 0:
            print('last step loss: {:.5f}'.format(loss.item()))
        print('load time:', time_load)
        print('run time:', time_run)

    def _save_checkpoint(self, epoch):
        save_path = os.path.join(self.save_dir, 'model.pt')
        torch.save(self.model, save_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            epoch_start = time.time()
            self._run_epoch(epoch)
            if self.gpu_id == 0:
                print('epoch time:', time.time() - epoch_start)
                if (epoch + 1) % self.save_every == 0:
                    self._save_checkpoint(epoch)


########################################################################
# parallelism

def setup(rank, world_size):
    # may need to change
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_train_objs(args):
    ckpt = f'/fs/nexus-scratch/rhaworth/models/progen/{args.model}' #f'./checkpoints/{args.model}'

    # load model + tokenizer
    model = create_model(ckpt=ckpt, fp16=args.fp16)
    tokenizer = create_tokenizer_custom(file='tokenizer.json')

    # load dataset
    def make_dataloader(dataset):
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=args.bsz,
                                           pin_memory=True,
                                           shuffle=False,
                                           sampler=DistributedSampler(dataset))
    
    train_dataset = ProteinBindingData(args.train, tokenizer, max_samples=args.max_samples)
    train_dataloader = make_dataloader(train_dataset)

    print('train samples found:', len(train_dataset))

    # configure training

    # default settings from https://huggingface.co/docs/transformers/v4.46.2/en/training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = args.n_epoch
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
            name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
            )
    
    loss_fn = torch.nn.CrossEntropyLoss()

    return train_dataloader, model, optimizer, lr_scheduler, loss_fn


########################################################################
# main

def main(rank, world_size, args):
    # parallelize
    setup(rank, world_size)

    # deterministic hopefully
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    # load everything
    train_dataloader, model, optimizer, lr_scheduler, loss_fn = load_train_objs(args)
    trainer = Trainer(model, train_dataloader, optimizer, lr_scheduler, loss_fn, rank, args.n_epoch, args.save)

    # full training run + save model
    trainer.train(args.n_epoch)

    # deparallelize
    cleanup()


if __name__ == '__main__':
    # don't bother supporting cpu parallelism
    if not torch.cuda.is_available():
        raise RuntimeError('No GPUs found')

    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B
    
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-medium')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true')) # unused
    parser.add_argument('--train', type=str, default='./data/uniprot_sprot_with_binding.tsv')
    parser.add_argument('--eval', type=str, default='') # unused
    parser.add_argument('--save', type=str, default='./weights')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=200)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)

    print('done.')
