#!/usr/bin/env python3
"""
pack_run.py — zip the most recent GUI agent run archive to /tmp/agent_run_latest.zip

Usage:
  python pack_run.py              # pack most recent run
  python pack_run.py list         # list all saved runs
  python pack_run.py <run_name>   # pack a specific run by name prefix
"""

import os
import sys
import zipfile
import json

RUNS_DIR = os.path.expanduser("~/.agent_bin/runs")
OUT_PATH = "/tmp/agent_run_latest.zip"


def list_runs():
    if not os.path.isdir(RUNS_DIR):
        print("No runs directory found at ~/.agent_bin/runs")
        return []
    runs = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        full = os.path.join(RUNS_DIR, name)
        if name == "latest" or os.path.islink(full):
            continue
        if not os.path.isdir(full):
            continue
        summary_path = os.path.join(full, "summary.json")
        meta = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    meta = json.load(f)
            except Exception:
                pass
        size_mb = sum(
            os.path.getsize(os.path.join(r, fn))
            for r, _, files in os.walk(full)
            for fn in files
        ) / 1_048_576
        runs.append((name, meta, size_mb))
    return runs


def find_run(prefix=None):
    """Return path to the run directory to pack."""
    if not os.path.isdir(RUNS_DIR):
        return None, None

    if prefix:
        # Find first run whose name starts with the prefix
        for name in sorted(os.listdir(RUNS_DIR), reverse=True):
            full = os.path.join(RUNS_DIR, name)
            if name.startswith(prefix) and os.path.isdir(full) and not os.path.islink(full):
                return full, name
        return None, None

    # Most recent: follow latest symlink, or pick newest dir by mtime
    latest_link = os.path.join(RUNS_DIR, "latest")
    if os.path.islink(latest_link):
        target = os.path.realpath(latest_link)
        if os.path.isdir(target):
            return target, os.path.basename(target)

    # Fall back: newest real directory by mtime
    dirs = [
        (os.path.join(RUNS_DIR, n), n)
        for n in os.listdir(RUNS_DIR)
        if n != "latest" and os.path.isdir(os.path.join(RUNS_DIR, n))
           and not os.path.islink(os.path.join(RUNS_DIR, n))
    ]
    if not dirs:
        return None, None
    return max(dirs, key=lambda p: os.path.getmtime(p[0]))


def pack(run_dir, run_name):
    print(f"Packing: {run_dir}")
    print(f"Output:  {OUT_PATH}")

    # Show summary if available
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                m = json.load(f)
            status = "SUCCESS" if m.get("success") else "FAILED"
            print(f"Status:  {status} — {m.get('iterations', '?')} iterations, "
                  f"{m.get('screenshots', '?')} screenshots")
            print(f"Task:    {m.get('task', '?')[:80]}")
        except Exception:
            pass

    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        file_count = 0
        for root, _, files in os.walk(run_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                arcname = os.path.join(run_name, os.path.relpath(fpath, run_dir))
                zf.write(fpath, arcname)
                file_count += 1

    size_mb = os.path.getsize(OUT_PATH) / 1_048_576
    print(f"Done: {file_count} files → {OUT_PATH} ({size_mb:.1f} MB)")
    return True


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        runs = list_runs()
        if not runs:
            print("No runs found.")
            return
        print(f"{'Run name':<45}  {'Status':<8}  {'Iters':<6}  {'Shots':<6}  {'MB':>5}")
        print("-" * 80)
        for name, meta, size_mb in runs:
            status = ("OK " if meta.get("success") else "FAIL") if meta else "   ?"
            iters  = str(meta.get("iterations", "?"))
            shots  = str(meta.get("screenshots", "?"))
            print(f"{name:<45}  {status:<8}  {iters:<6}  {shots:<6}  {size_mb:>5.1f}")
        return

    prefix = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir, run_name = find_run(prefix)

    if not run_dir:
        if prefix:
            print(f"No run found matching prefix: {prefix}")
        else:
            print("No runs found in ~/.agent_bin/runs")
        sys.exit(1)

    pack(run_dir, run_name)


if __name__ == "__main__":
    main()
