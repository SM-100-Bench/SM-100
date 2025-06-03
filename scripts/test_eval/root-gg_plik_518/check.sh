#!/bin/bash
if grep -Fq 'out.Close()' server/data/file/file.go; then
    exit 0
else
    exit 1
fi