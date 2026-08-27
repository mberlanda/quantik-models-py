"""Batched AlphaZero MCTS: one tree per game, descents vectorized across games.

The usual laptop bottleneck for a Python AlphaZero is that every simulation
walks its tree in interpreted code. Here the `g` games in flight descend in
lockstep instead: at each tree level the PUCT scores for all active games are
computed as one `(g, 64)` NumPy op, so a simulation round costs a fixed
handful of array ops regardless of how many games are running, and every
leaf in the round reaches the network in a single batch.

Trees live in one flat arena of `(max_nodes, 64)` edge statistics. The arena
is rebuilt per move rather than re-rooted, which keeps it small enough
(`games * simulations + games` nodes) to stay resident.

Sign convention matches `fastboard`: a value is always from the perspective
of the side to move at that node, and both terminal conditions are a loss
for the side to move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..env import fastboard as fb
from .evaluator import Evaluator

MAX_PLIES = 16
_UNEXPANDED = np.int32(-1)


@dataclass(frozen=True)
class MCTSParams:
    simulations: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.6
    dirichlet_weight: float = 0.25
    # First-play-urgency reduction: an unvisited edge inherits its parent's Q
    # minus this, so a node whose explored children all lose still prefers a
    # fresh sibling. 0.0 reproduces plain AlphaZero (unvisited Q = 0).
    fpu_reduction: float = 0.2


class BatchedMCTS:
    """Runs `simulations` PUCT simulations for a batch of live positions."""

    def __init__(self, evaluator: Evaluator, params: MCTSParams, rng: np.random.Generator):
        self.evaluator = evaluator
        self.params = params
        self.rng = rng

    def search(
        self, boards: npt.NDArray[np.uint16], add_noise: bool = True
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return `(visit_counts, root_values)` for a batch of live boards.

        `visit_counts` is `(g, 64)` float32 edge visits at each root;
        `root_values` is the mean backed-up value at each root, from that
        root's side-to-move perspective.
        """
        g = boards.shape[0]
        if g == 0:
            return (
                np.zeros((0, fb.ACTION_COUNT), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        sims = self.params.simulations
        capacity = g * (sims + 2)

        # --- arena -------------------------------------------------------
        node_bb = np.zeros((capacity, 8), dtype=np.uint16)
        node_legal = np.zeros((capacity, fb.ACTION_COUNT), dtype=np.bool_)
        node_terminal = np.zeros(capacity, dtype=np.bool_)
        child = np.full((capacity, fb.ACTION_COUNT), _UNEXPANDED, dtype=np.int32)
        edge_n = np.zeros((capacity, fb.ACTION_COUNT), dtype=np.float32)
        edge_w = np.zeros((capacity, fb.ACTION_COUNT), dtype=np.float32)
        edge_p = np.zeros((capacity, fb.ACTION_COUNT), dtype=np.float32)
        node_value = np.zeros(capacity, dtype=np.float32)

        roots = np.arange(g, dtype=np.int32)
        node_bb[roots] = boards
        node_legal[roots] = fb.legal_masks(boards)
        priors, values = self.evaluator(boards, node_legal[roots])
        edge_p[roots] = priors
        node_value[roots] = values
        used = g

        if add_noise and self.params.dirichlet_weight > 0.0:
            edge_p[roots] = self._noisy(edge_p[roots], node_legal[roots])

        for _ in range(sims):
            path_node, path_action, depth, leaf_parent, leaf_action, live = self._descend(
                roots, child, node_legal, node_terminal, edge_n, edge_w, edge_p, node_value
            )
            leaf_value, used = self._expand(
                leaf_parent,
                leaf_action,
                live,
                node_bb,
                node_legal,
                node_terminal,
                node_value,
                child,
                edge_p,
                used,
            )
            self._backup(path_node, path_action, depth, leaf_value, live, edge_n, edge_w)

        visits = edge_n[roots].copy()
        total = visits.sum(axis=1)
        root_values = np.where(
            total > 0, edge_w[roots].sum(axis=1) / np.maximum(total, 1.0), node_value[roots]
        ).astype(np.float32)
        return visits, root_values

    # -- internals --------------------------------------------------------

    def _noisy(
        self, priors: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float32]:
        """Blend Dirichlet noise into the root priors, over legal actions only."""
        weight = self.params.dirichlet_weight
        out = priors.copy()
        for i in range(priors.shape[0]):
            idx = np.flatnonzero(legal[i])
            if idx.size == 0:
                continue
            noise = self.rng.dirichlet(np.full(idx.size, self.params.dirichlet_alpha))
            out[i, idx] = (1.0 - weight) * priors[i, idx] + weight * noise
        return out

    def _descend(
        self, roots, child, node_legal, node_terminal, edge_n, edge_w, edge_p, node_value
    ):
        """Walk every game from its root to a leaf edge, one level at a time.

        Returns the visited `(node, action)` path per game, its length, the
        edge that needs expanding, and which games are still descending.
        """
        g = roots.shape[0]
        c_puct = self.params.c_puct
        fpu = self.params.fpu_reduction

        path_node = np.zeros((MAX_PLIES + 1, g), dtype=np.int32)
        path_action = np.zeros((MAX_PLIES + 1, g), dtype=np.int64)
        depth = np.zeros(g, dtype=np.int64)
        cur = roots.copy()
        live = np.ones(g, dtype=np.bool_)
        leaf_parent = np.full(g, -1, dtype=np.int32)
        leaf_action = np.zeros(g, dtype=np.int64)

        for level in range(MAX_PLIES + 1):
            idx = np.flatnonzero(live)
            if idx.size == 0:
                break
            nodes = cur[idx]
            n = edge_n[nodes]
            w = edge_w[nodes]
            p = edge_p[nodes]
            legal = node_legal[nodes]

            visited = n > 0
            total = n.sum(axis=1, keepdims=True)
            # Parent Q from the parent's own perspective; a child's stored W is
            # already signed that way, so the FPU prior is a direct subtraction.
            parent_q = np.where(
                total > 0, w.sum(axis=1, keepdims=True) / np.maximum(total, 1.0), 0.0
            )
            q = np.where(visited, w / np.maximum(n, 1.0), parent_q - fpu)
            u = c_puct * p * np.sqrt(total) / (1.0 + n)
            score = np.where(legal, q + u, -np.inf)
            action = score.argmax(axis=1)

            path_node[level, idx] = nodes
            path_action[level, idx] = action
            depth[idx] = level + 1

            nxt = child[nodes, action]
            needs_expansion = nxt == _UNEXPANDED
            # Stop where the edge is unexplored (expand it) or leads to a
            # terminal node (its value is known without a network call).
            stop = needs_expansion.copy()
            explored = ~needs_expansion
            if explored.any():
                stop[explored] |= node_terminal[nxt[explored]]

            leaf_parent[idx] = nodes
            leaf_action[idx] = action
            cur[idx] = np.where(needs_expansion, nodes, nxt)

            live[idx[stop]] = False
            if not live.any():
                break

        return path_node, path_action, depth, leaf_parent, leaf_action, np.ones(g, dtype=np.bool_)

    def _expand(
        self,
        leaf_parent,
        leaf_action,
        live,
        node_bb,
        node_legal,
        node_terminal,
        node_value,
        child,
        edge_p,
        used,
    ):
        """Create or look up each game's leaf node; return its value."""
        g = leaf_parent.shape[0]
        rows = np.arange(g)
        existing = child[leaf_parent, leaf_action]
        fresh = existing == _UNEXPANDED

        leaf_value = np.zeros(g, dtype=np.float32)
        # Already-materialized leaves are terminal by construction of _descend.
        old = np.flatnonzero(~fresh)
        if old.size:
            leaf_value[old] = node_value[existing[old]]

        new = np.flatnonzero(fresh)
        if new.size:
            ids = np.arange(used, used + new.size, dtype=np.int32)
            used += new.size
            boards = fb.apply_actions(node_bb[leaf_parent[new]], leaf_action[new])
            node_bb[ids] = boards
            legal = fb.legal_masks(boards)
            node_legal[ids] = legal
            done = fb.has_winning_line(boards) | ~legal.any(axis=1)
            node_terminal[ids] = done
            child[leaf_parent[new], leaf_action[new]] = ids

            node_value[ids[done]] = -1.0
            open_ = np.flatnonzero(~done)
            if open_.size:
                open_ids = ids[open_]
                priors, values = self.evaluator(boards[open_], legal[open_])
                edge_p[open_ids] = priors
                node_value[open_ids] = values
            leaf_value[new] = node_value[ids]
        del rows, live
        return leaf_value, used

    @staticmethod
    def _backup(path_node, path_action, depth, leaf_value, live, edge_n, edge_w):
        """Add the leaf value back along each path, flipping sign per ply."""
        g = depth.shape[0]
        max_depth = int(depth.max()) if g else 0
        for level in range(max_depth):
            idx = np.flatnonzero(depth > level)
            if idx.size == 0:
                continue
            nodes = path_node[level, idx]
            actions = path_action[level, idx]
            # The mover at `level` matches the leaf's mover when the number of
            # plies between them is even.
            sign = np.where((depth[idx] - level) % 2 == 0, 1.0, -1.0).astype(np.float32)
            np.add.at(edge_n, (nodes, actions), 1.0)
            np.add.at(edge_w, (nodes, actions), leaf_value[idx] * sign)
        del live
