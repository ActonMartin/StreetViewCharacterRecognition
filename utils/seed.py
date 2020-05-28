import torch


def init_seeds(seed=0):
    # sets the seed for generating random numbers.
    torch.manual_seed(seed)
    # Sets the seed for generating random numbers for the current GPU.
    # It’s safe to call this function if CUDA is not available; in that case,
    # it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs.
    # It’s safe to call this function if CUDA is not available;
    # in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
