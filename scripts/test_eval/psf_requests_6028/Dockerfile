FROM python:3.8

WORKDIR /repo

ENV PIP_INDEX_URL=https://pypi.bismuth.cloud/20210802/ 

CMD ["bash", "-c", "make && exec pytest -- tests/test_utils.py"]