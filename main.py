"""Thin wrapper. Prefer: python scripts/train.py --task mixed"""
import subprocess
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    subprocess.run([sys.executable, "scripts/train.py"] + args)