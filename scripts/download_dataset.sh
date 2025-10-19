#!/bin/bash

TASK_NAMES=("task-0000" "task-0001" "task-0023" "task-0035" "task-0047")

# for DIR in annotations data skill_prompts videos meta/episodes; do

for TASK_NAME in "${TASK_NAMES[@]}"; do
    for DIR in skill_prompts; do
        echo "Syncing ${DIR} for ${TASK_NAME}..."
        aws s3 sync \
            s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/ \
            /vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/
    done
done

# This is done separately because it is not task-specific. In fact, it is redundant except for the
# first time that any task is downloaded to a local machine.
aws s3 sync --exclude "episodes/*" \
    s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/meta/ \
    /vision/group/behavior/2025-challenge-demos/meta/
