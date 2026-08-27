"""Probe: how expensive is an exact Quantik solve from various plies?"""
import time, random
from quantik_core import State
from quantik_core.minimax import MinimaxEngine, MinimaxConfig
from quantik_core.move import generate_legal_moves_list, apply_move
from quantik_core.qfen import bb_to_qfen

EMPTY = State.from_qfen("..../..../..../....")

def random_position(plies, rng):
    bb = EMPTY.bb
    for _ in range(plies):
        moves = generate_legal_moves_list(bb)
        if not moves:
            return None
        bb = apply_move(bb, rng.choice(moves))
    return State(bb)

rng = random.Random(7)
for plies in (8, 6, 5, 4, 3, 2, 1, 0):
    times, nodes, scores = [], [], []
    trials = 3 if plies <= 2 else 5
    for t in range(trials):
        st = random_position(plies, rng) if plies else EMPTY
        if st is None:
            continue
        from quantik_core.game_utils import has_winning_line
        if has_winning_line(st.bb) or not generate_legal_moves_list(st.bb):
            continue
        eng = MinimaxEngine(MinimaxConfig(max_depth=16, time_limit_s=None))
        t0 = time.perf_counter()
        res = eng.solve(st)
        dt = time.perf_counter() - t0
        times.append(dt); nodes.append(res.nodes); scores.append(res.score)
        if plies == 0:
            print(f"  EMPTY BOARD solve: {dt:.2f}s nodes={res.nodes} score={res.score} best={res.best_move}")
    if times:
        print(f"ply={plies}: n={len(times)} mean={sum(times)/len(times):.3f}s max={max(times):.3f}s "
              f"mean_nodes={sum(nodes)//len(nodes)} scores={[round(s,1) for s in scores]}", flush=True)
