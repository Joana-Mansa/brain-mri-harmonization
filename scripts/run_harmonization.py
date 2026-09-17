"""Fit harmonizers inside the shared experiment; never fit to the full cohort."""

import sys
from brain_harmonization.cli import main

if __name__ == "__main__":
    main(["analyze", *sys.argv[1:]])
