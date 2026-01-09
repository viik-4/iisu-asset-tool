#!/usr/bin/env python3
"""
CLI for iiSU Icon Generator
Uses the consolidated run_backend.py module
"""
import argparse
import signal
import sys
from pathlib import Path

import yaml

import run_backend


# Graceful stop handling
STOP_REQUESTED = False

def _handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[STOP] Stop requested — finishing in-flight items, then exiting...\n")

signal.signal(signal.SIGINT, _handle_sigint)


def parse_args():
    p = argparse.ArgumentParser(description="iiSU Icon Generator - CLI")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    p.add_argument("--platform", default="", help="Comma-separated platform keys (e.g. NES,PS2). Empty = all configured.")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    p.add_argument("--limit", type=int, default=0, help="Limit titles per platform (0 = use config or unlimited)")
    p.add_argument("--mode", default="", help="Source mode: steamgriddb_then_libretro, steamgriddb, libretro, libretro_then_steamgriddb (empty = use config)")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    # Load config to get platforms
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[ERROR] Failed to read config: {e}")
        return 1

    platforms_cfg = cfg.get("platforms", {}) or {}
    if not platforms_cfg:
        print("[ERROR] No platforms configured in config.yaml")
        return 1

    # Platform selection
    if args.platform:
        selected = [x.strip() for x in args.platform.split(",") if x.strip()]
        platforms = [k for k in selected if k in platforms_cfg]
        if not platforms:
            print(f"[ERROR] No matching platforms found for --platform={args.platform}")
            print(f"Available: {', '.join(sorted(platforms_cfg.keys()))}")
            return 1
    else:
        platforms = sorted(platforms_cfg.keys())

    print(f"[CONFIG] {config_path}")
    print(f"[PLATFORMS] {', '.join(platforms)}")
    print(f"[WORKERS] {args.workers}")
    if args.limit > 0:
        print(f"[LIMIT] {args.limit} per platform")
    print()

    # Create cancel token
    cancel = run_backend.CancelToken()

    # Simple CLI callbacks
    def log_cb(msg):
        if not STOP_REQUESTED:
            print(msg)

    def progress_cb(done, total):
        if total > 0 and done % max(1, total // 20) == 0:  # Log every ~5%
            pct = int((done / total) * 100)
            print(f"[PROGRESS] {done}/{total} ({pct}%)")

    callbacks = {
        "log": log_cb,
        "progress": progress_cb,
    }

    # Check for stop request periodically
    import threading
    def check_stop():
        if STOP_REQUESTED:
            cancel.cancel()

    stop_checker = threading.Timer(0.5, check_stop)
    stop_checker.daemon = True
    stop_checker.start()

    # Run job
    try:
        ok, msg = run_backend.run_job(
            config_path=config_path,
            platforms=platforms,
            workers=args.workers,
            limit=args.limit,
            cancel=cancel,
            callbacks=callbacks,
            source_mode=args.mode if args.mode else None,
            steamgriddb_square_only=None,  # Use config default
        )

        print()
        print("[RESULT]", msg)
        return 0 if ok else 1

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n[ERROR] Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
