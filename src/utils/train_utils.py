import random
import numpy as np
import torch
import logging
def worker_init_fn(worker_id):
    """Ensures DataLoader workers use deterministic seeding"""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )