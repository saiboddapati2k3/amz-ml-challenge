"""Run a module with peak-RSS logging:  python scripts/peakmem.py src.block --split test ...

Samples this process's RSS every 0.5 s and prints the peak on exit (also on crash)."""
import os
import runpy
import sys
import threading
import time

import psutil

sys.path.insert(0, os.getcwd())
proc, peak = psutil.Process(), [0]


def _watch():
    while True:
        peak[0] = max(peak[0], proc.memory_info().rss)
        time.sleep(0.5)


threading.Thread(target=_watch, daemon=True).start()
mod, sys.argv = sys.argv[1], [sys.argv[1]] + sys.argv[2:]
t0 = time.time()
try:
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
finally:
    print(f"peak RSS {peak[0] / 2**30:.2f} GB, wall {time.time() - t0:.0f}s", flush=True)
