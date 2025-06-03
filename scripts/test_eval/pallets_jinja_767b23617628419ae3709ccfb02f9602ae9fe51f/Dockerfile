FROM python:3.12

WORKDIR /repo

CMD ["bash", "-c", "pip install -r requirements/tests.txt && pip install -e . && exec pytest"]