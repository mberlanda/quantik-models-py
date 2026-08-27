import time
import numpy as np
from quantik_models.env import fastboard as fb

rng = np.random.default_rng(0)
for n in (256, 2048):
    bb = fb.empty_boards(n)
    t0 = time.perf_counter(); plies = 0; steps = 0
    alive = np.arange(n)
    cur = bb
    while len(cur):
        done, _ = fb.terminal_status(cur)
        cur = cur[~done]
        if not len(cur): break
        m = fb.legal_masks(cur)
        # uniform random legal action
        r = rng.random((len(cur), 64)) * m
        a = r.argmax(axis=1)
        cur = fb.apply_actions(cur, a)
        plies += len(cur); steps += 1
    dt = time.perf_counter() - t0
    print(f"batch={n}: {n/dt:.0f} games/s, {plies/dt:.0f} plies/s ({steps} vector steps, {dt*1000:.1f} ms)")

# raw legal_masks throughput
bb = fb.empty_boards(4096)
for _ in range(5):
    m = fb.legal_masks(bb); r = rng.random((4096,64))*m; bb = fb.apply_actions(bb, r.argmax(1))
t0=time.perf_counter(); k=0
while time.perf_counter()-t0 < 1.0:
    fb.legal_masks(bb); k+=1
print(f"legal_masks: {k*4096/(time.perf_counter()-t0):.0f} pos/s")
t0=time.perf_counter(); k=0
while time.perf_counter()-t0 < 1.0:
    fb.encode_tensors(bb); k+=1
print(f"encode_tensors: {k*4096/(time.perf_counter()-t0):.0f} pos/s")
