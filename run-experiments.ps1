python -m venv env
.\env\Scripts\activate
pip install -U wheel setuptools
pip install -r .\requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
$env:PYTHONPATH = '.'
python .\src\main.py