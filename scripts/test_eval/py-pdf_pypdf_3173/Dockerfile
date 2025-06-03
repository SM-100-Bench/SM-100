FROM python:3.12

WORKDIR /repo

CMD ["bash", "-c", "pip install -e .[image,dev] && exec pytest -- tests/generic/test_image_inline.py"]