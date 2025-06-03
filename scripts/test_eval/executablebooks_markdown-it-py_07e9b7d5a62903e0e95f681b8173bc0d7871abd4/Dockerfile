FROM python:3.10

WORKDIR /repo

CMD ["bash", "-c", "pip install -e .[testing,linkify,benchmarking] && exec pytest"]