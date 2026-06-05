"""
Enables running ftmwpipeline as a module with python -m ftmwpipeline
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
