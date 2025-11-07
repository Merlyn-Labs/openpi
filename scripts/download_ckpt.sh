#!/bin/bash

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20251029_024836/25000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251029_024836/25000/
