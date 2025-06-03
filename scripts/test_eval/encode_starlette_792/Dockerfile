FROM python:3.8

WORKDIR /repo

ENV PIP_INDEX_URL=https://pypi.bismuth.cloud/20200217/

CMD ["bash", "-c", "pip install -r requirements.txt && pip install -e . && exec pytest -W ignore::DeprecationWarning"]