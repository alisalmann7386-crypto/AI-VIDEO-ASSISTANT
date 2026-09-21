"""Legacy test entry point. Real offline regression tests are in tests/."""

if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]))
