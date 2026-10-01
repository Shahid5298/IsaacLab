import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
ea = EventAccumulator(run, size_guidance={'scalars': 0})
ea.Reload()
tags = ea.Tags()['scalars']
print("=== TAGS ===")
for t in sorted(tags):
    print(" ", t)
print()
for t in sorted(tags):
    ev = ea.Scalars(t)
    if not ev: continue
    # print sampled values
    n = len(ev)
    idxs = sorted(set([0, n//8, n//4, 3*n//8, n//2, 5*n//8, 3*n//4, 7*n//8, n-1]))
    vals = [(ev[i].step, ev[i].value) for i in idxs]
    mx = max(ev, key=lambda e: e.value)
    print(f"--- {t}  (n={n}) max={mx.value:.5g}@{mx.step}")
    print("   ", "  ".join(f"{s}:{v:.4g}" for s,v in vals))
