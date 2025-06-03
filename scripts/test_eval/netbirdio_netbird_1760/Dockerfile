FROM golang:1.24

RUN apt-get update && \
    apt-get install -y libpcap-dev

WORKDIR /repo

CMD ["go", "test", "./client/internal/..."]