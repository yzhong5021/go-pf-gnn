import numpy as np

from tests.generate_toy_data import generate_toy_data, random_sequence


def test_random_seq_len() -> None:
    np.random.seed(0)
    seq = random_sequence(min_len=5, max_len=5)
    assert len(seq) == 5


def test_data_shapes() -> None:
    df, vocab, labels = generate_toy_data(
        n_samples=4,
        n_go_terms=3,
        min_len=4,
        max_len=6,
        seed=7,
    )

    assert len(df) == 4
    assert len(vocab) == 3
    assert labels.shape == (4, 3)
