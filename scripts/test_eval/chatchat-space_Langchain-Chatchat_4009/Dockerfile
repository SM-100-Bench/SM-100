FROM python:3.10

COPY test.py /

WORKDIR /repo

CMD ["bash", "-c", "pip install langchain langchain_community fastapi torch chardet && python3 copy_config_example.py && python3 /test.py"]