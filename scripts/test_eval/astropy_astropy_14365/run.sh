#!/bin/bash
set -xeuo pipefail

source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
sed -i 's/requires = \["setuptools",/requires = \["setuptools==68.0.0",/' pyproject.toml
python -m pip install -e .[test] --verbose
exec pytest --no-header -rA --tb=no -p no:cacheprovider