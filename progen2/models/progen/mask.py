# methods to generate flexible causal masks from sequences
import numpy as np

# helper function: convert list of indices to list of contiguous segments
# represented as list of tuples (start, end)
def idx_to_segments(idx):
    segments = []
    seg_start = idx[0]
    prev = seg_start
    # iterate over elements past first
    for i in idx[1:]:
        if i - prev == 1:
            # expand segment
            prev = i
        else:
            # complete segment
            segments.append((seg_start, prev))
            # start new segment
            seg_start = i
            prev = i
    # add last segment
    segments.append((seg_start, idx[-1]))

    return segments

# OLD random path to construct sequence starting from known indices
# TODO: optimize; likely to become bottleneck during training, especially in batch
def idx_to_path(idx, seqlen):
    path = []
    idx_curr = sorted(idx)
    while len(idx_curr) < seqlen:
        # get list of contiguous segments in idx_curr as 2-tuples
        # single-element segments have same value for each tuple field, this is fine
        segments = idx_to_segments(idx_curr)
        
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

# updated idx to path for n_lm_head=2, also yielding targets
def idx_to_path_targets(idx, seqlen, dim=512):
    path = []
    targets = np.ones((dim, 2), dtype=int)
    targets *= -100
    idx = sorted(idx)

    assert idx[0] >= 0
    assert idx[-1] < seqlen

    # find all binding site segments
    segments = idx_to_segments(idx)

    # find largest segment, use to set next_L and next_R for path
    max_seg_idx = np.argmax([seg[1] - seg[0] for seg in segments])
    max_seg = segments[max_seg_idx]

    next_L = max_seg[0] - 1
    next_R = max_seg[1] + 1

    # set targets for binding site
    for seg in segments:
        prev = seg[0] - 1
        next = seg[1] + 1
        targets[seg[0], 0] = prev if prev != -1 else -100
        targets[seg[1], 1] = next if next != seqlen else -100

    # convert idx to set for faster membership checking
    idx = set(idx)

    # iterate until BOS + EOS
    while next_L != -1 or next_R != seqlen:
        # are L and R both valid?
        choices = []
        choices.append(next_L) if next_L != -1 else None
        choices.append(next_R) if next_R != seqlen else None

        # select move, update path
        step = np.random.choice(choices)
        path.append(step)
        idx.add(step)

        # if we picked next_L, update next_L
        if step == next_L:
            next_L -= 1
            # check for collisions
            if next_L in idx:
                # assume we have few enough segments that binary search isn't worth doing
                for seg in segments:
                    # next_L should collide with the end of a segment
                    if next_L == seg[1]:
                        next_L = seg[0] - 1
                        break
        # otherwise, update next_R
        else:
            next_R += 1
            # check for collisions in opposite direction
            if next_R in idx:
                for seg in segments:
                    if next_R == seg[0]:
                        next_R = seg[1] + 1
        
        # update targets
        targets[step, 0] = step - 1 if step - 1 != -1 else -100
        targets[step, 1] = step + 1 if step + 1 != seqlen else -100
    
    #full_idxs = np.arange(seqlen)
    #targets[full_idxs[1:], 0] = full_idxs[:-1]
    #targets[full_idxs[:-1], 1] = full_idxs[1:]

    return path, targets

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
    
    #path = idx_to_path(idx, seqlen)
    path, targets = idx_to_path_targets(idx, seqlen, dim)
    mask = path_to_mask(path, idx, dim)

    # old target generation
    '''
    # default: ignore all
    targets = np.ones(dim, dtype=int)
    targets *= -100 
    # set targets from path; output is seq indices, must convert to token ids externally
    targets[idx] = path[0]
    for i in range(len(path)-1):
        targets[path[i]] = path[i+1]
    '''

    # TODO: min(idx) is not the offset we should be using anymore, fix later
    #       for now, we don't even use offset, so leave it
    return mask, np.min(idx), targets

# generate random path through sequence of known length
# just_binding arg: skip making the mask, just return the binding site
def rand_mask_start(seqlen, dim=512, exp_sz=5, p_drop=0.2, just_binding=False):
    # generate artificial binding site position
    sz = max(1, np.random.poisson(exp_sz))
    keep_idx = np.random.random(sz) > p_drop
    if np.sum(keep_idx) == 0:
        keep_idx[0] = True
    start = np.random.randint(0, seqlen-sz)
    idx = np.arange(start, start+sz)[keep_idx]
    
    if just_binding:
        return idx

    # get mask and start position
    return idx_to_mask_start(idx, seqlen, dim)