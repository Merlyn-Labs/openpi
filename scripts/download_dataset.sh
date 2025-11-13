#!/bin/bash

aws s3 sync s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/meta/diverse_prompts.jsonl /vision/group/behavior/2025-challenge-demos/meta/diverse_prompts.jsonl

# This is done separately because it is not task-specific. In fact, it is redundant except for the
# first time that any task is downloaded to a local machine.
aws s3 sync --exclude "episodes/*" \
    s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/meta/ \
    /vision/group/behavior/2025-challenge-demos/meta/

hf download \
    behavior-1k/2025-challenge-demos \
    --repo-type dataset \
    --include annotations/ \
    --local-dir /vision/group/behavior/2025-challenge-demos/

# # for DIR in annotations data skill_prompts videos meta/episodes; do
# for i in $(seq 0 49); do
#     TASK_NAME=$(printf "task-%04d" "$i")
#     echo "Syncing ${TASK_NAME}..."
#     for DIR in skill_prompts; do
#         echo "Syncing ${DIR} for ${TASK_NAME}..."
#         aws s3 sync \
#             s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/ \
#             /vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/
#     done
# done
