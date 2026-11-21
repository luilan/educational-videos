"""A slow check (3 seconds), like a linter or a test suite. Usage: python3 check.py lint|types|tests"""
import sys
import time

time.sleep(3)
print({"lint": "lint: 0 problems", "types": "types: 0 errors", "tests": "tests: 14 passed"}.get(sys.argv[1], "unknown check"))
