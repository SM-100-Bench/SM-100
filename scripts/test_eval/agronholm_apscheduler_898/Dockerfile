FROM python:3.12

WORKDIR /repo

CMD ["bash", "-c", "cd /repo && python -m venv /tmp/venv && source /tmp/venv/bin/activate && pip install --upgrade pip && pip install -e .[test] && pytest -- tests/triggers"]