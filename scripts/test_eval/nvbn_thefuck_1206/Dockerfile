FROM python:3.10

WORKDIR /repo

CMD ["bash", "-c", "pip install -Ur requirements.txt && python setup.py develop && exec pytest tests/rules/test_dirty_unzip.py"]