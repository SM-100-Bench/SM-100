#!/bin/bash
if grep -Fq 'delete(s.containerInitExit' cmd/containerd-shim-runc-v2/task/service.go; then
    exit 0
else
    exit 1
fi