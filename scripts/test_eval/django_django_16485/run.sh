#!/bin/bash
set -xeuo pipefail

source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
python -m pip install -e .
exec ./tests/runtests.py --verbosity 2