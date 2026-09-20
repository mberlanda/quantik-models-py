# Corpus structure: what the rows actually are

Corpora are quoted by row count. The row count describes the **value** corpus and
badly misdescribes the **policy** corpus: about 8% of rows carry a policy label,
the labels sit at a fixed number per ply, and nothing reaches below ply 3. This
document measures all of that. It changes no corpus.

Every number below was produced by the command printed beside it, run from the
root of `quantik-models-py` against `runs/oracle/corpus/` (gitignored, so this
file is what survives the directory) on 2026-09-20. Related:
[`corpora.md`](corpora.md), [`corpus-v3.md`](corpus-v3.md),
[`labeling-strategy.md`](labeling-strategy.md).

## Measuring

One script produces every table in this document. Save it as `measure.py`
(anywhere) and run `.venv/bin/python measure.py` from the repository root.

```python
import numpy as np

FILES = [("v1", "exact-sampled.npz"), ("v2", "exact-sampled-v2.npz"),
         ("v3", "exact-sampled-v3.npz")]
for name, f in FILES:
    z = np.load("runs/oracle/corpus/" + f)
    print(name, {k: (z[k].dtype.name, z[k].shape) for k in z.files})
    pl = z["plies"]
    # a row is policy-labelled iff its weight (v1) or its mask (v2/v3) is nonzero
    lab = (z["policy_weight"] > 0) if "policy_weight" in z.files else (z["optimal_mask"] != 0)
    print(name, "rows", len(pl), "labelled", int(lab.sum()),
          "density %.4f" % lab.mean(), "minply", int(pl.min()), "maxply", int(pl.max()))
    print(name, "bytes/row", sum(z[k].nbytes for k in z.files) / len(pl),
          {k: z[k].nbytes // len(pl) for k in z.files})
    for p in range(int(pl.min()), int(pl.max()) + 1):
        m = pl == p
        print(f"  ply {p:2d} rows {int(m.sum()):8d} labelled {int(lab[m].sum()):7d}")
```

## Policy density

| corpus | rows | policy-labelled | density |
| --- | --- | --- | --- |
| v1 (`exact-sampled.npz`) | 3,087,356 | 250,000 | 8.10% |
| v2 (`exact-sampled-v2.npz`) | 3,196,958 | 255,058 | 7.98% |
| v3 (`exact-sampled-v3.npz`) | 3,520,526 | 271,676 | 7.72% |

Density is flat and, if anything, falls. v3 has 323,568 more rows than v2 but
only 16,618 more policy labels (271,676 against 255,058): a corpus that "grew
by 323,568 rows" grew its policy signal by about a twentieth of that. The other ~92% are value-only rows: they train the value head and are
masked out of the policy loss.

## The per-ply table: the cap is visible

Rows / policy-labelled rows per ply, from the same script. Read the labelled
column, not the row column.

| ply | v1 rows | v1 labelled | v2 rows | v2 labelled | v3 rows | v3 labelled |
| --- | --- | --- | --- | --- | --- | --- |
| 0-2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 664 | 664 | 726 | 726 |
| 4 | 0 | 0 | 9,664 | 957 | 9,758 | 983 |
| 5 | 0 | 0 | 22,655 | 1,728 | 29,905 | 4,899 |
| 6 | 40,000 | 40,000 | 86,631 | 41,709 | 170,766 | 55,068 |
| 7 | 846,816 | 60,000 | 876,804 | 60,000 | 1,108,831 | 60,000 |
| 8 | 1,001,185 | 60,000 | 1,001,185 | 60,000 | 1,001,185 | 60,000 |
| 9 | 698,460 | 30,000 | 698,460 | 30,000 | 698,460 | 30,000 |
| 10 | 255,278 | 20,000 | 255,278 | 20,000 | 255,278 | 20,000 |
| 11 | 118,055 | 20,000 | 118,055 | 20,000 | 118,055 | 20,000 |
| 12 | 86,741 | 20,000 | 86,741 | 20,000 | 86,741 | 20,000 |
| 13 | 40,821 | 0 | 40,821 | 0 | 40,821 | 0 |

At plies 7-12 the labelled count is `60,000 / 60,000 / 30,000 / 20,000 / 20,000 /
20,000` in **all three** corpora, whatever the row count is (ply 7 has 846,816,
876,804 or 1,108,831 rows and exactly 60,000 labels in each). Round numbers
that do not move when the rows do are a sampling budget, not coverage. Ply 13
has none at all. At ply 8, 60,000 labels over 1,001,185 rows is 6.0%; at ply 12,
20,000 over 86,741 is 23.1%: the fraction is whatever the cap leaves over.

The shallow plies are the opposite case. There the labelled count *does* move
(ply 6: 40,000, then 41,709, then 55,068), and plies 3-6 are exactly where the
three corpora differ. Note also that at plies 4 and 5 the labelled share is
under 17% even in v3.

## The two policy schemas and their byte costs

The same script prints per-array byte costs (`bytes/row` lines):

| | v1 | v2 / v3 |
| --- | --- | --- |
| `boards` | `uint16 (N, 8)`, 16 B | `uint16 (N, 8)`, 16 B |
| `value_target` | `float32 (N,)`, 4 B | `float32 (N,)`, 4 B |
| `plies` | `int16 (N,)`, 2 B | `int16 (N,)`, 2 B |
| policy label | `policy_target float32 (N, 64)`, 256 B, plus `policy_weight float32 (N,)`, 4 B | `optimal_mask uint64 (N,)`, 8 B |
| **total** | **282 B/row** | **30 B/row** |

Measured totals are exactly 282.0 and 30.0 bytes/row. The policy label alone is
260 B against 8 B, 32.5 times. The mask is a bitset of optimal `action_index`
values (bit `i` set means action `i` is optimal).

Whether dense to mask loses anything depends on whether the dense rows are
uniform over their support. Checked over all 250,000 labelled v1 rows:

```python
import numpy as np
z = np.load("runs/oracle/corpus/exact-sampled.npz")
l = z["policy_weight"] > 0
t = z["policy_target"][l]
n = (t > 0).sum(1)
print("mean opt %.4f max %d" % (n.mean(), n.max()),
      "rowsum max dev", abs(t.sum(1) - 1).max(),
      "uniform dev", abs(t - (t > 0) / n[:, None]).max(),
      "weights", np.unique(z["policy_weight"]))
```

```
mean opt 4.2199 max 31 rowsum max dev 1.1920929e-07 uniform dev 9.934107481068821e-09 weights [0. 1.]
```

Every labelled row is uniform over its support to float32 precision, and the
weight is only ever 0 or 1, so v1's dense row carries no information the mask
does not (mean 4.22 optimal moves, max 31). For the mask files the same count is
`np.unpackbits(m[m != 0].view(np.uint8)).reshape(-1, 64).sum(1)`: v2 mean 4.43,
max 43; v3 mean 4.78, max 43. Those means are not comparable with v1's, because
the labelled rows come from different plies.

## The ply floor

No corpus contains a row below ply 3: v1's minimum ply is 6, v2's and v3's are
3 (the per-ply table above; `pl.min()` in the script). The canonical live
positions at plies 0, 1 and 2 are 1, 3 and 51. The command below prints the
counts for levels 1, 2 and 3 (ply 0 is the empty board, so it has no level file):

```
.venv/bin/python -c "import numpy as np; print([np.load(f'runs/canonical/level0{i}.npy', mmap_mode='r').shape[0] for i in (1,2,3)])"
[3, 51, 726]
```

`runs/canonical/counts.json` records `"0": {"live": 1}`, so plies 0-2 are 1 + 3 + 51 = **55 canonical positions**, and no model trained on
any of these corpora has seen one.

Ply 3 has 726 canonical positions and v3 holds 726 rows there, so v3 is the
first corpus with the complete ply-3 level (v2's 664 is 62 short). The
rows-per-ply column for plies 4-6 against the canonical live counts
(10,946; 105,632; 901,916) shows how partial the rest is: v3 holds 89%, 28% and
19% of those levels.

## What the published figures got right and wrong

Confirmed by measurement: the totals (3,087,356 / 3,196,958 / 3,520,526), the
labelled counts (250,000 / 255,058 / 271,676), the 60k/60k/30k/20k/20k/20k
labels at plies 7-12 in all three corpora, zero labels at ply 13, the 282 versus
30 bytes/row, and the 55 canonical positions at plies 0-2.

Corrected or sharpened: the density is 8.10% / 7.98% / 7.72%, not a flat "~8%",
so it drifts down as rows are added. The cap is a property of plies 7-12; ply 6
is not capped in v1 (all 40,000 rows labelled) and not in v2/v3. `corpora.md`'s
"mean 4.22, max 31 optimal moves" is v1 only.

No corpus file was modified: the script only reads.
