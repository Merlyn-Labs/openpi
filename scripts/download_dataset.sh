#!/bin/bash

TASK_NAME="task-0000"

for DIR in annotations data skill_prompts videos meta/episodes; do
    echo "Syncing ${DIR}..."
    aws s3 sync \
        s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/ \
        /vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/
done

aws s3 sync --exclude "episodes/*" \
    s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/meta/ \
    /vision/group/behavior/2025-challenge-demos/meta/
