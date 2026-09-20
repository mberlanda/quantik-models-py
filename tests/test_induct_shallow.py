"""The completeness guard and the induction of plies 0-2 (QW-027/W3).

The real-corpus tests need ``runs/`` (gitignored) and skip on CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantik_models.data.exact_corpus import ExactCorpus
from quantik_models.env import fastboard as fb
from scripts import induct_shallow as ish

ROOT = Path(__file__).resolve().parent.parent
# QUANTIK_RUNS_DIR lets a git worktree (which has no runs/) point at the main checkout's.
RUNS = Path(os.environ.get("QUANTIK_RUNS_DIR", ROOT / "runs"))
V3 = RUNS / "oracle/corpus/exact-sampled-v3.npz"
LEVEL03 = RUNS / "canonical/level03.npy"
needs_runs = pytest.mark.skipif(not (V3.exists() and LEVEL03.exists()), reason="needs gitignored runs/ corpora")


def _level3_from_engine() -> np.ndarray:
    return ish._canonical_children(ish.shallow_levels()[2])


def _synthetic_ply3_corpus(level3: np.ndarray, rng: np.random.Generator) -> ExactCorpus:
    n = len(level3)
    return ExactCorpus(
        boards=level3,
        optimal_mask=np.ones(n, np.uint64),
        value_target=np.where(rng.random(n) < 0.5, 1.0, -1.0).astype(np.float32),
        plies=np.full(n, 3, np.int16),
    )


def test_shallow_levels_have_the_canonical_sizes():
    assert [len(x) for x in ish.shallow_levels()] == [1, 3, 51]
    assert len(_level3_from_engine()) == 726


def test_incomplete_level_aborts_naming_the_missing_keys():
    level3 = _level3_from_engine()
    corpus = _synthetic_ply3_corpus(level3, np.random.default_rng(0))
    dropped = fb.canonical_keys(level3[[7, 300]])
    keep = np.setdiff1d(np.arange(len(level3)), [7, 300])
    short = ExactCorpus(corpus.boards[keep], corpus.optimal_mask[keep], corpus.value_target[keep], corpus.plies[keep])
    with pytest.raises(ish.IncompleteLevelError) as info:
        ish.induct_shallow(short, level3)
    for key in dropped:
        assert f"{int(key):#018x}" in str(info.value)


def test_value_only_ply3_row_aborts_naming_its_key():
    level3 = _level3_from_engine()
    corpus = _synthetic_ply3_corpus(level3, np.random.default_rng(5))
    corpus.optimal_mask[42] = 0
    with pytest.raises(ish.IncompleteLevelError, match="value-only") as info:
        ish.induct_shallow(corpus, level3)
    assert f"{int(fb.canonical_keys(level3[[42]])[0]):#018x}" in str(info.value)


def test_equal_count_but_different_set_still_aborts():
    """The reason the check is a set comparison: swap one row for a duplicate, count unchanged."""
    level3 = _level3_from_engine()
    swapped = level3.copy()
    swapped[5] = swapped[6]
    assert len(swapped) == len(level3)
    corpus = _synthetic_ply3_corpus(swapped, np.random.default_rng(1))
    with pytest.raises(ish.IncompleteLevelError, match="missing"):
        ish.induct_shallow(corpus, level3)


def test_a_non_canonical_extra_position_aborts():
    level3 = _level3_from_engine()
    other = ish.shallow_levels()[2][:1]  # a ply-2 board masquerading as ply 3
    boards = np.concatenate([level3[:-1], other])
    corpus = _synthetic_ply3_corpus(boards, np.random.default_rng(2))
    with pytest.raises(ish.IncompleteLevelError):
        ish.induct_shallow(corpus, level3)


def test_complete_synthetic_level_yields_55_positions_and_nonempty_masks():
    level3 = _level3_from_engine()
    corpus = _synthetic_ply3_corpus(level3, np.random.default_rng(3))
    out, counts = ish.induct_shallow(corpus, level3)
    assert counts == {0: 1, 1: 3, 2: 51}
    assert len(out) == 55
    assert np.all(out.optimal_mask != 0)
    assert np.all(np.abs(out.value_target) == 1.0)


def test_main_reports_abort_and_writes_nothing(tmp_path, capsys):
    level3 = _level3_from_engine()
    corpus = _synthetic_ply3_corpus(level3[:-3], np.random.default_rng(4))
    corpus_path = corpus.save(tmp_path / "short.npz")
    np.save(tmp_path / "level03.npy", level3)
    out = tmp_path / "out.npz"
    rc = ish.main(["--corpus", str(corpus_path), "--level03", str(tmp_path / "level03.npy"), "--out", str(out)])
    assert rc == 2 and not out.exists()
    assert "missing" in capsys.readouterr().err


@needs_runs
def test_real_v3_is_complete_and_inducts_55_positions():
    corpus = ExactCorpus.load(V3)
    out, counts = ish.induct_shallow(corpus, np.load(LEVEL03))
    assert counts == {0: 1, 1: 3, 2: 51}
    assert np.all(out.optimal_mask != 0)
    # the empty board: every move is legal and, by symmetry, the induced set is a valid subset
    root = out.optimal_mask[out.plies == 0][0]
    assert int(root) != 0
