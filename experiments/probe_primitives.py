import time, random
import numpy as np
from quantik_core import State
from quantik_core.move import generate_legal_moves_list, apply_move
from quantik_core.game_utils import has_winning_line
from quantik_core.ml_data import qfen_to_tensor
from quantik_core.qfen import bb_to_qfen

EMPTY = State.from_qfen("..../..../..../....")
rng = random.Random(0)

# random playout throughput
t0=time.perf_counter(); games=0; plies=0
while time.perf_counter()-t0 < 2.0:
    bb = EMPTY.bb
    while True:
        if has_winning_line(bb): break
        mv = generate_legal_moves_list(bb)
        if not mv: break
        bb = apply_move(bb, rng.choice(mv)); plies+=1
    games+=1
dt=time.perf_counter()-t0
print(f"random playouts: {games/dt:.0f} games/s, {plies/dt:.0f} plies/s")

# movegen alone
bb = EMPTY.bb
for _ in range(4): bb = apply_move(bb, rng.choice(generate_legal_moves_list(bb)))
t0=time.perf_counter(); n=0
while time.perf_counter()-t0 < 1.0:
    generate_legal_moves_list(bb); n+=1
print(f"movegen: {n/(time.perf_counter()-t0):.0f}/s")

# qfen + tensor
q = bb_to_qfen(bb)
t0=time.perf_counter(); n=0
while time.perf_counter()-t0 < 1.0:
    qfen_to_tensor(q, 0); n+=1
print(f"qfen_to_tensor: {n/(time.perf_counter()-t0):.0f}/s")

# torch fwd
import torch
from quantik_models.model.policy_value_net import PolicyValueNet, PRESETS, parameter_count
for dev in ("cpu","mps"):
    m = PolicyValueNet(PRESETS["small"]).to(dev).eval()
    for bs in (1, 64, 512):
        x = torch.zeros(bs,9,4,4, device=dev)
        with torch.no_grad():
            for _ in range(3): m(x)
            if dev=="mps": torch.mps.synchronize()
            t0=time.perf_counter(); k=0
            while time.perf_counter()-t0 < 0.7:
                m(x); k+=1
            if dev=="mps": torch.mps.synchronize()
            d=time.perf_counter()-t0
        print(f"  {dev} bs={bs}: {k/d:.0f} fwd/s -> {k*bs/d:.0f} pos/s")
print("small params:", parameter_count(PolicyValueNet(PRESETS['small'])))
