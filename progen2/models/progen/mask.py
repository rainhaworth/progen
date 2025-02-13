# methods to generate flexible causal masks from sequences
import numpy as np

# random path to construct sequence starting from known indices
# TODO: optimize; likely to become bottleneck during training, especially in batch
def idx_to_path(idx, seqlen):
    path = []
    idx_curr = sorted(idx)
    while len(idx_curr) < seqlen:
        # get list of contiguous segments in idx_curr as 2-tuples
        # single-element segments have same value for each tuple field, this is fine
        segments = []
        seg_start = idx_curr[0]
        prev = seg_start
        # iterate over elements past first; hopefully it just skips this and doesn't throw an error if 1 element
        for idx in idx_curr[1:]:
            if idx - prev == 1:
                prev = idx
                continue
            else:
                # complete segment
                segments.append((seg_start, prev))
                # start new segment
                seg_start = idx
                prev = idx
        # get last segment
        segments.append((seg_start, idx_curr[-1]))
        
        # check whether we need to move further past the leftmost and rightmost elements in the sequence
        found_start = (idx_curr[0] == 0)
        found_end = (idx_curr[-1] == seqlen - 1)
        
        # for each segment, there are 2 possible steps: move left of leftmost index or move right of rightmost
        # get list of all such steps
        steps = []
        for segment in segments:
            steps.append(segment[0]-1)
            steps.append(segment[1]+1)

        # prune if necessary
        if found_start:
            steps = steps[1:]
        if found_end:
            steps = steps[:-1]
        
        # get unique; if 2 segments are separated by 1 index, that index will be duplicated and its probability will be doubled
        # this is probably not a huge deal but i would rather avoid doing it unintentionally
        steps = np.unique(steps)
        
        # select move, update path and idx
        step = np.random.choice(steps)
        path.append(step)
        idx_curr.append(step)
        idx_curr = sorted(idx_curr)
    
    return path

# from path, i.e. sequence of indices representing steps, and indices of known monomers, generate mask
def path_to_mask(path, idx, dim=512):
    mask = np.zeros((dim, dim))

    # for each index in original binding site, unmask the entire column
    mask[:len(path)+len(idx), idx] = 1

    # construct mask
    for path_idx, step in enumerate(path):
        # for this step + all later path steps, add on the current step
        populated_idxs = path[path_idx:]
        mask[populated_idxs, step] = 1

    return mask

# from known indices and sequence length, generate mask and return binding site start position
# new: also generate targets
def idx_to_mask_start(idx, seqlen, dim=512):
    assert len(idx) <= seqlen
    assert seqlen <= dim
    
    path = idx_to_path(idx, seqlen)
    mask = path_to_mask(path, idx, dim)

    # default: ignore all
    targets = np.ones(dim, dtype=int)
    targets *= -100 
    # set targets from path; output is seq indices, must convert to token ids externally
    targets[idx] = path[0]
    for i in range(len(path)-1):
        targets[path[i]] = path[i+1]

    return mask, np.min(idx), targets

# generate random path through sequence of known length
def rand_mask_start(seqlen, dim=512, exp_sz=5, p_drop=0.2):
    # generate artificial binding site position
    sz = max(1, np.random.poisson(exp_sz))
    keep_idx = np.random.random(sz) > p_drop
    if np.sum(keep_idx) == 0:
        keep_idx[0] = True
    start = np.random.randint(0, seqlen-sz)
    idx = np.arange(start, start+sz)[keep_idx]
    
    # get mask and start position
    return idx_to_mask_start(idx, seqlen, dim)
