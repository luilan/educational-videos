"""Run all tests (and a basic style check). Usage: python3 run_tests.py"""
import subprocess
import sys

lines = open("invoice.py").read().splitlines()
long = [i + 1 for i, line in enumerate(lines) if len(line) > 120]
if long:
    sys.exit(f"style: lines longer than 120 characters: {long}")
sys.exit(subprocess.run([sys.executable, "test_invoice.py"]).returncode)
