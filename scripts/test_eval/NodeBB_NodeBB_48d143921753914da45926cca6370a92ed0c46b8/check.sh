#!/bin/bash
if grep -Fq 'const Namespaces = {};' src/socket.io/index.js; then
    exit 1
else
    exit 0
fi