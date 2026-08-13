"""
Participant-disjoint train/test split, frozen to a JSON config file so
every later experiment is comparable against the exact same split
(ActionPlan.md Priority 0 / Step 4).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from config import DEFAULT_TEST_PARTICIPANT_IDS, VALIDATION_FRACTION
from preprocessing.io import iter_raw_instances


def discover_participant_ids(raw_root: Path) -> list[str]:
    ids = sorted({inst.participant_id for inst in iter_raw_instances(raw_root)})
    return ids


def assert_disjoint_splits(split: dict) -> None:
    """Guarantee train/val/test participant sets never overlap.

    This should be structurally impossible given how build_or_load_split()
    constructs the split, but a frozen split.json could in principle be
    hand-edited or produced by a future code path that doesn't preserve the
    invariant -- so we verify it explicitly every time a split is loaded or
    built, rather than assuming it. Raises immediately on any leakage
    instead of letting it silently inflate downstream metrics.
    """
    train_ids, val_ids, test_ids = set(split["train"]), set(split["val"]), set(split["test"])
    overlaps = {
        "train/val": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "val/test": val_ids & test_ids,
    }
    bad = {pair: sorted(ids) for pair, ids in overlaps.items() if ids}
    if bad:
        raise RuntimeError(f"Participant split leakage detected: {bad}")
    print(
        f"[split] leakage check passed: train={len(train_ids)} "
        f"val={len(val_ids)} test={len(test_ids)} -- all disjoint"
    )


def build_or_load_split(
    raw_root: Path,
    split_path: Path,
    seed: int = 42,
    test_fraction: float = 0.18,
    force_rebuild: bool = False,
) -> dict:
    """Return {"train": [...], "val": [...], "test": [...]} participant IDs.

    If split_path already exists, it is loaded as-is (frozen). Otherwise a
    new split is built: DEFAULT_TEST_PARTICIPANT_IDS are used as the test
    set wherever they exist in the raw data (matching the original
    prototype's exact test IDs for continuity); any remaining participants
    needed to reach test_fraction are added via a seeded random draw so the
    split is still fully reproducible. A validation slice is carved out of
    the remaining train participants (participant-disjoint from both).

    Either way, the returned split is verified disjoint before being handed
    back to the caller (see assert_disjoint_splits).
    """
    split_path = Path(split_path)
    if split_path.exists() and not force_rebuild:
        with open(split_path) as f:
            split = json.load(f)
        assert_disjoint_splits(split)
        return split

    all_ids = discover_participant_ids(raw_root)
    if not all_ids:
        raise RuntimeError(f"No participants found under {raw_root}")

    rng = random.Random(seed)
    target_test_n = max(1, round(len(all_ids) * test_fraction))

    test_ids = [pid for pid in DEFAULT_TEST_PARTICIPANT_IDS if pid in all_ids]
    remaining = [pid for pid in all_ids if pid not in test_ids]
    rng.shuffle(remaining)
    while len(test_ids) < target_test_n and remaining:
        test_ids.append(remaining.pop())
    test_ids = sorted(set(test_ids))

    trainval_ids = [pid for pid in all_ids if pid not in test_ids]
    rng.shuffle(trainval_ids)
    val_n = max(1, round(len(trainval_ids) * VALIDATION_FRACTION))
    val_ids = sorted(trainval_ids[:val_n])
    train_ids = sorted(trainval_ids[val_n:])

    split = {"train": train_ids, "val": val_ids, "test": test_ids, "seed": seed}
    assert_disjoint_splits(split)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    return split