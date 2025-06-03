#!/bin/bash
if grep -Fq 'if (parent' src/components/Modal.js || grep -Fq 'if (!parent' src/components/Modal.js; then
    exit 0
else
    exit 1
fi