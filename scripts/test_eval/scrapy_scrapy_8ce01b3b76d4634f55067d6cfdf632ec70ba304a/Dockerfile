FROM python:3.10

WORKDIR /repo

# Limit to tests/test_downloadermiddleware_cookies.py because things are slow
CMD ["bash", "-c", "export PIP_INDEX_URL=https://pypi.bismuth.cloud/20220301/ && pip install -r tests/requirements.txt && pip install -e . && exec pytest --durations=10 -- tests/test_downloadermiddleware_cookies.py"]