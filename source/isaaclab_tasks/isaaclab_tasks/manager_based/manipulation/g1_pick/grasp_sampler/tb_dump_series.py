import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
tags = sys.argv[2:]
ea = EventAccumulator(run, size_guidance={'scalars': 0})
ea.Reload()
for t in tags:
    ev = ea.Scalars(t)
    n = len(ev)
    idxs = list(range(0, n, max(1, n // 40)))
    print(f"--- {t} ---")
    print(" ".join(f"{ev[i].step}:{ev[i].value:.4g}" for i in idxs))
