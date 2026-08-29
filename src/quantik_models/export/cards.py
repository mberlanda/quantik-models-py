"""Prose and diagrams the model cards carry, kept out of the generator.

Separate from `huggingface.py` because this is editorial content — the
project's own description of itself — and the generator is mechanism. Mixing
them makes it hard to see which parts of a card are derived from a file and
which are written.

The architecture diagrams live here rather than only in the repository README
because a Hub repo is read on its own, with no README beside it. A test
asserts every registered architecture has one, so a new architecture cannot
ship a card that silently omits its own diagram.
"""

from __future__ import annotations

PROJECT = """\
## About this project

Quantik began as a holiday rivalry and became an engineering project. Before
building an AI to play — or teach — the game, the game itself had to be
represented precisely: an exact notation, a canonical form under the board's
192 symmetries, and a bitboard the rules can be computed on cheaply.

That foundation is what these models are trained on. Every label is exact,
produced by a solver rather than by self-play, so the network fits ground
truth instead of its own earlier opinions. The engineering is written up as a
series on **The Full-Stack Mind**: first-principles representation, then
Monte-Carlo search, beam search, exact endgame proof, and a tournament where
the engines finally played each other.

- Series: <https://mauroberlanda.substack.com/t/quantik>
"""

# One per registry name. Rendered natively by the Hub.
DIAGRAMS: dict[str, str] = {
    "resnet": """\
```mermaid
flowchart LR
  IN["board<br/>(B,9,4,4)"] --> STEM["stem<br/>Conv3x3 9→C · BN · ReLU"]
  STEM --> TRUNK["trunk<br/>B × residual block<br/>Conv3x3 · BN · ReLU · Conv3x3 · BN · +skip"]
  TRUNK --> PH["policy head<br/>Conv1x1 C→2 · flatten · Linear 32→64"]
  TRUNK --> VH["value head<br/>Conv1x1 C→1 · flatten · Linear · tanh"]
  PH --> POL["policy logits (B,64)"]
  VH --> VAL["value (B,)"]
```""",
    "mlp": """\
```mermaid
flowchart LR
  IN["board<br/>(B,9,4,4)"] --> FL["flatten → 144"]
  FL --> STEM["Linear 144→H · BN · ReLU"]
  STEM --> TRUNK["trunk<br/>B × dense residual block<br/>Linear · BN · ReLU · Linear · BN · +skip"]
  TRUNK --> PH["policy head<br/>Linear H→64"]
  TRUNK --> VH["value head<br/>Linear H→64 · ReLU · Linear · tanh"]
  PH --> POL["policy logits (B,64)"]
  VH --> VAL["value (B,)"]
```""",
    "cpool": """\
```mermaid
flowchart LR
  IN["board<br/>(B,9,4,4)"] --> STEM["stem<br/>Linear 9→C per cell · LayerNorm"]
  STEM --> BLK
  subgraph BLK["constraint block × B"]
    direction LR
    P["pool over the 12 groups<br/>4 rows · 4 columns · 4 zones"] --> M["mix<br/>Linear · GELU"]
    M --> S["scatter back to cells"]
    S --> R["+ residual · LayerNorm"]
  end
  BLK --> PH["policy head<br/>Linear C→4 per cell<br/>transpose → 64"]
  BLK --> VH["value head<br/>mean over cells · MLP · tanh"]
  PH --> POL["policy logits (B,64)"]
  VH --> VAL["value (B,)"]
```""",
    "attn": """\
```mermaid
flowchart LR
  IN["board<br/>(B,9,4,4)"] --> TOK["16 cell tokens<br/>Linear 9→D + learned position"]
  TOK --> BLK
  subgraph BLK["pre-norm encoder block × B"]
    direction LR
    N1["LayerNorm"] --> MHA["multi-head self-attention"]
    MHA --> R1["+ residual"]
    R1 --> N2["LayerNorm"] --> FF["FFN"] --> R2["+ residual"]
  end
  BLK --> PH["policy head<br/>Linear D→4 per cell<br/>transpose → 64"]
  BLK --> VH["value head<br/>mean over cells · MLP · tanh"]
  PH --> POL["policy logits (B,64)"]
  VH --> VAL["value (B,)"]
```""",
}

# What each architecture is *for*. One sentence, because a card is read by
# someone deciding whether to download it, not by someone auditing it.
SUMMARY: dict[str, str] = {
    "resnet": "A convolutional residual trunk — the incumbent design, and the "
    "one every hyperparameter in this project was originally chosen for.",
    "mlp": "A dense control that throws spatial structure away entirely. It "
    "exists to make \"convolution is worth having on a 4x4 board\" falsifiable "
    "rather than assumed.",
    "cpool": "Pools over Quantik's twelve constraint groups — four rows, four "
    "columns, four 2x2 zones — so the rule that decides the game is wired into "
    "the architecture rather than learned from data.",
    "attn": "A self-attention encoder over the sixteen cells. The same bet as "
    "the constraint model without the prior: it is told nothing about rows, "
    "columns or zones and has to discover them.",
}


def diagram_for(arch: str) -> str:
    return DIAGRAMS.get(arch, "")


def summary_for(arch: str) -> str:
    return SUMMARY.get(arch, "")
