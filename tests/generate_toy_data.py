"""
generate_toy_data.py

Generates data for toy implementation.
Only two columns are needed: sequences and GO terms.
These will be used to create embeddings and to generate the functional graph.
"""

import numpy as np
import pandas as pd
import random


AA = list("ACDEFGHIKLMNPQRSTVWY")


def random_sequence(min_len, max_len):
    length = random.randint(min_len, max_len)
    return "".join(random.choice(AA) for _ in range(length))


def generate_toy_data(n_samples=100, n_go_terms=10, min_len=30, max_len=60, seed=1000):
    """
    Create a toy dataset with sequences and GO term annotations.

    Args:
        n_samples (int): sample size
        n_go_terms (int): size of mock GO vocabulary
        min_len (int): minimum protein length
        max_len (int): maximum protein length
        seed (int): random seed

    Returns:
        df with columns ["sequence", "go_terms"]
        go_vocab: list of GO term IDs used.
        labels: multi-hot label array [n_samples, n_go_terms]
    """

    random.seed(seed)
    np.random.seed(seed)

    go_vocab = [f"GO:{i:07d}" for i in range(1, n_go_terms+1)]

    data = []
    labels = np.zeros((n_samples, n_go_terms), dtype=int)

    for i in range(n_samples):
        seq = random_sequence(min_len=min_len, max_len=max_len)
        # assign each protein 1–3 random GO terms
        k = random.randint(1, 3)
        terms = random.sample(go_vocab, k)
        data.append({"sequence": seq, "go_terms": terms})

        # multi-hot encoding
        for t in terms:
            labels[i, go_vocab.index(t)] = 1

    df = pd.DataFrame(data)
    return df, go_vocab, labels