from __future__ import annotations

import numpy as np


def split_match_ids(
    match_ids: list[int],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> dict[str, list[int]]:
    if not match_ids:
        return {"train": [], "val": [], "test": []}

    uniq = sorted(set(int(m) for m in match_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)

    n_train = max(1, min(n, int(np.floor(train_frac * n))))
    remaining = n - n_train
    if remaining <= 0:
        return {"train": uniq, "val": [], "test": []}

    val_plus_test = max(val_frac + test_frac, 1e-9)
    val_share = val_frac / val_plus_test
    n_val = int(round(val_share * remaining))
    n_val = max(0, min(remaining, n_val))
    n_test = remaining - n_val

    train_ids = uniq[:n_train]
    val_ids = uniq[n_train : n_train + n_val]
    test_ids = uniq[n_train + n_val :]

    return {"train": train_ids, "val": val_ids, "test": test_ids}
