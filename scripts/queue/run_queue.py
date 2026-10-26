#!/usr/bin/env python
"""Run queued GPU jobs one after another so the GPU never idles.

A job is a shell script in outputs/queue/pending/ ; jobs run in name order (prefix them 010_, 020_, ...).
Each job's stdout/stderr goes to outputs/queue/logs/<job>.log; the script then moves to done/ or failed/.
The runner polls for new jobs and exits when outputs/queue/STOP exists.

    nohup python scripts/queue/run_queue.py &
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime

from swm.paths import OUTPUTS, ROOT

Q = OUTPUTS / "queue"


def main() -> None:
    for d in ("pending", "running", "done", "failed", "logs"):
        (Q / d).mkdir(parents=True, exist_ok=True)
    while not (Q / "STOP").exists():
        jobs = sorted((Q / "pending").glob("*.sh"))
        if not jobs:
            time.sleep(30)
            continue
        job = jobs[0]
        run = Q / "running" / job.name
        job.rename(run)
        log = Q / "logs" / f"{job.stem}.log"
        start = datetime.now()
        with log.open("a") as fh:
            fh.write(f"### start {start.isoformat(timespec='seconds')}\n")
            fh.flush()
            rc = subprocess.run(["bash", str(run)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
            fh.write(f"### end {datetime.now().isoformat(timespec='seconds')} rc={rc}\n")
        run.rename(Q / ("done" if rc == 0 else "failed") / job.name)
        with (Q / "history.tsv").open("a") as fh:
            fh.write(f"{start.isoformat(timespec='seconds')}\t{datetime.now().isoformat(timespec='seconds')}\t"
                     f"{job.name}\t{rc}\n")


if __name__ == "__main__":
    main()
