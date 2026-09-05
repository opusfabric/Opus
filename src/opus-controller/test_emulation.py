#!/usr/bin/env python3
"""Run a self-contained smoke test for the controller's emulation worker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def exchange(path: Path, value: int) -> str:
    for _ in range(200):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2.0)
                client.connect(str(path))
                client.sendall(str(value).encode())
                return client.recv(256).decode().strip()
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.01)
    raise RuntimeError(f"timed out waiting for {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delay-ms", type=int, default=50,
        help="emulated topology-change delay in milliseconds (default: 50)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config = root / "src" / "opus-controller" / "config.py"
    demo_dir = Path(tempfile.mkdtemp(prefix="opus-controller-smoke-"))
    prefix = "opus_smoke"
    socket_path = demo_dir / f"{prefix}_0"
    log_path = demo_dir / "controller.log"
    environment = os.environ.copy()
    environment.update(OPUS_IPC_DIR=str(demo_dir), OPUS_IPC_PREFIX=prefix)

    worker = None
    try:
        with log_path.open("w") as log:
            worker = subprocess.Popen(
                [sys.executable, "-u", str(config), "-e", str(args.delay_ms), "1", "0"],
                cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
            )

        for value in (0, 1, 3, 1):
            if worker.poll() is not None:
                raise RuntimeError(
                    f"controller exited with status {worker.returncode}; see {log_path}"
                )
            print(value, exchange(socket_path, value))

        print("controller emulation smoke test: PASS")
        return 0
    except Exception as error:
        print(f"controller emulation smoke test: FAIL: {error}", file=sys.stderr)
        if log_path.exists():
            print(log_path.read_text(), file=sys.stderr, end="")
        return 1
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        shutil.rmtree(demo_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
