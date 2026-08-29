#!/usr/bin/env python3
"""Verify accentedge-benchmark installation."""
import sys


def check(name, fn):
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        sys.exit(1)


def main():
    print("AccentEdge Benchmark v1 - Installation Check")
    print("=" * 50)
    checks = [
        ("schemas importable", lambda: __import__("accentedge_benchmark.schemas")),
        ("audio io importable", lambda: __import__("accentedge_benchmark.audio.io")),
        ("numpy available", lambda: __import__("numpy")),
        ("pydantic available", lambda: __import__("pydantic")),
        ("soundfile available", lambda: __import__("soundfile")),
        ("librosa available", lambda: __import__("librosa")),
        ("pyyaml available", lambda: __import__("yaml")),
        ("jinja2 available", lambda: __import__("jinja2")),
        ("jiwer available", lambda: __import__("jiwer")),
    ]
    for name, fn in checks:
        check(name, fn)
    print("=" * 50)
    print("All checks passed.")


if __name__ == "__main__":
    main()
