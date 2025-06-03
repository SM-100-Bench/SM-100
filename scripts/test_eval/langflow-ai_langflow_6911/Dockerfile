FROM python:3.12

WORKDIR /repo

CMD ["bash", "-c", "curl -LsSf https://astral.sh/uv/0.6.17/install.sh | sh && export PATH=/root/.local/bin:$PATH && uv sync && make unit_tests"]