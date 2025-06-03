FROM python:3.8

WORKDIR /repo

ENV PIP_INDEX_URL=https://pypi.bismuth.cloud/20210120/

CMD ["bash", "-c", "pip install -r requirements-dev.txt && exec ./scripts/run_tests.sh"]