#!/bin/bash
python -m venv env
source env/bin/activate
pip install -U wheel setuptools
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
PYTHONPATH=. python ./src/main.py