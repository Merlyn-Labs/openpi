#!/bin/bash

TASK_NAMES=(
    "task-0000"
    "task-0001"
    "task-0005"
    "task-0015"
    "task-0016"
    "task-0017"
    "task-0018"
    "task-0022"
    "task-0023"
    "task-0030"
    "task-0034"
    "task-0035"
    "task-0037"
    "task-0038"
    "task-0039"
    "task-0040"
    "task-0045"
    "task-0046"
    "task-0047"
)

# for DIR in annotations data skill_prompts videos meta/episodes; do

for TASK_NAME in "${TASK_NAMES[@]}"; do
    for DIR in skill_prompts; do
        echo "Syncing ${DIR} for ${TASK_NAME}..."
        aws s3 sync \
            s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/ \
            /scratch/vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/
    done
done

aws s3 sync s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/meta/diverse_prompts.jsonl /scratch/vision/group/behavior/2025-challenge-demos/meta/diverse_prompts.jsonl

# # This is done separately because it is not task-specific. In fact, it is redundant except for the
# # first time that any task is downloaded to a local machine.
# aws s3 sync --exclude "episodes/*" \
#     s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/meta/ \
#     /vision/group/behavior/2025-challenge-demos/meta/
