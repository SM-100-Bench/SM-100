FROM golang:1.24

WORKDIR /repo

CMD ["bash", "-c", "CGO_ENABLED=0 exec go test -v -tags kqueue,dev ./..."]
