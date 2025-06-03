#!/bin/bash
if grep -Fq 'if (!prop) {' packages/core/src/utils/serializeNode.tsx; then
    exit 1
else
    exit 0
fi