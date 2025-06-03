#!/bin/bash
if grep -Fq 'eval(' lollms/functions/calculator.py; then
    exit 1
else
    exit 0
fi