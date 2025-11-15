#!/usr/bin/env python3
"""
Script to extract all unique skill_description entries from annotation JSON files.
"""

import json
import glob
from pathlib import Path
from typing import Set, Dict
from collections import Counter
import traceback

def load_task_prompts(tasks_jsonl_path: str) -> Dict[int, Dict[str, str]]:
    """
    Load task information from tasks.jsonl file.
    
    Args:
        tasks_jsonl_path: Path to the tasks.jsonl file
        
    Returns:
        Dictionary mapping task_index to task information (task_name, task prompt)
    """
    tasks = {}
    with open(tasks_jsonl_path, 'r') as f:
        for line in f:
            task_data = json.loads(line.strip())
            tasks[task_data['task_index']] = {
                'task_name': task_data['task_name'],
                'task': task_data['task']
            }
    return tasks

def extract_skill_descriptions_with_frames(annotations_dir: str) -> tuple[Set[str], Dict[str, int], Dict[int, int]]:
    """
    Read all JSON files in the annotations directory and extract unique skill descriptions
    along with the total number of frames dedicated to each skill.
    
    Args:
        annotations_dir: Path to the annotations directory
        
    Returns:
        Tuple of (set of unique skill descriptions, dict mapping description to total frames, dict mapping task_idx to total frames)
    """
    index = 0
    unique_descriptions = set()
    frame_counts = {}
    skill_string_counts = Counter()
    task_frame_counts = {}  # Track frames per task

    # Find all JSON files matching the pattern
    # ACCEPTABLE_TASKS = [26, 15, 14, 42, 44, 36, 3, 25, 34, 6, 13, 40, 49, 16, 1, 2, 22, 8, 30, 38, 39, 0]
    ACCEPTABLE_TASKS = [1]
    json_files = []
    task_to_files = {}  # Track which files belong to which task
    for task_idx in ACCEPTABLE_TASKS:
        pattern = f"{annotations_dir}/task-{task_idx:04d}/*.json"
        files = glob.glob(pattern)
        json_files.extend(files)
        task_to_files[task_idx] = files
        task_frame_counts[task_idx] = 0  # Initialize frame count for this task

    print(f"Found {len(json_files)} JSON files to process")

    episode_lengths = []

    # Process each JSON file
    for file_index, json_file in enumerate(json_files):
        try:
            # Determine which task this file belongs to
            current_task_idx = None
            for task_idx, files in task_to_files.items():
                if json_file in files:
                    current_task_idx = task_idx
                    break
            
            with open(json_file, 'r') as f:
                data = json.load(f)

            task_duration = data['meta_data']['task_duration']
            episode_lengths.append(task_duration)

            # Extract skill descriptions from skill_annotation array
            if 'skill_annotation' in data:
                # print(f"File Index: {file_index}")
                # print(len(data['skill_annotation']))
                # full_task_skill_string = "–".join([f"{skill['skill_description'][0]}({', '.join(skill['object_id'][0])})" for skill in data['skill_annotation']])
                # if full_task_skill_string == "move to(door_bexenl_0)–open door(door_bexenl_0)–move to(storage_box_80)–pick up from(storage_box_80, floors_ulujpr_0)–move to(floors_nbxnpk_0)–place on(storage_box_80, floors_nbxnpk_0)–move to(storage_box_79)–pick up from(storage_box_79, floors_ulujpr_0)–move to(storage_box_80)–place on(storage_box_79, floors_nbxnpk_0)":
                #     print(f"PROBLEMATIC FILE: {json_file}")
                # skill_string_counts[full_task_skill_string] += 1
                for skill_idx, skill in enumerate(data['skill_annotation']):
                    # print(f"{skill_idx}: {skill['frame_duration'][0]}-{skill['frame_duration'][1]}: {skill['skill_description'][0]}({', '.join(skill['object_id'][0])})")
                    if 'skill_description' in skill:
                        # skill_description is a list, so iterate through it
                        assert len(skill['skill_description']) == 1, "Expected only one skill description per skill"
                        description = skill['skill_description'][0]
                        unique_descriptions.add(description)

                        # Calculate frames for this skill, handle both 2-number list or N lists of 2 numbers
                        if 'frame_duration' in skill:
                            fd = skill['frame_duration']
                            # If fd is exactly 2 numbers (not a list of lists)
                            if isinstance(fd, list) and len(fd) == 2 and all(isinstance(x, int) for x in fd):
                                start_frame, end_frame = fd
                                num_frames = end_frame - start_frame
                                frame_counts[description] = frame_counts.get(description, 0) + num_frames
                                if current_task_idx is not None:
                                    task_frame_counts[current_task_idx] += num_frames
                            # If fd is a list of lists (N pairs)
                            elif isinstance(fd, list) and all(isinstance(x, list) and len(x) == 2 for x in fd):
                                for pair in fd:
                                    start_frame, end_frame = pair
                                    num_frames = end_frame - start_frame
                                    frame_counts[description] = frame_counts.get(description, 0) + num_frames
                                    if current_task_idx is not None:
                                        task_frame_counts[current_task_idx] += num_frames
                            else:
                                raise ValueError(f"Unexpected frame_duration format: {fd}")

            index += 1
            if index % 100 == 0:
                print(f"Processed {index} files")

        except Exception as e:
            print(f"Error processing {json_file}: {traceback.format_exc()}")
            continue

    # for skill_string, count in skill_string_counts.items():
    #     print(f"{skill_string}: {count}")

    return unique_descriptions, frame_counts, episode_lengths, task_frame_counts, ACCEPTABLE_TASKS


def main():
    annotations_dir = "/vision/group/behavior/2025-challenge-demos/annotations"
    tasks_jsonl_path = "/vision/group/behavior/2025-challenge-demos/meta/tasks.jsonl"
    
    # Load task prompts
    print("Loading task information...")
    task_prompts = load_task_prompts(tasks_jsonl_path)
    
    print("Extracting unique skill descriptions...")
    unique_descriptions, frame_counts, episode_lengths, task_frame_counts, acceptable_tasks = extract_skill_descriptions_with_frames(annotations_dir)
    
    # Print task prompts
    print("\n" + "=" * 80)
    print("TASK PROMPTS FOR ACCEPTABLE_TASKS:")
    print("=" * 80)
    for task_idx in acceptable_tasks:
        if task_idx in task_prompts:
            task_info = task_prompts[task_idx]
            print(f"\nTask {task_idx}: {task_info['task_name']}")
            print(f"Prompt: {task_info['task']}")
    print("=" * 80)
    
    # Print per-task frame counts
    print("\nFRAME COUNTS PER TASK:")
    print("=" * 80)
    for task_idx in acceptable_tasks:
        task_name = task_prompts[task_idx]['task_name'] if task_idx in task_prompts else f"task-{task_idx}"
        frames = task_frame_counts.get(task_idx, 0)
        print(f"Task {task_idx:3d} ({task_name:40s}): {frames:>12,} frames")
    print("=" * 80)
    
    print(f"\nFound {len(unique_descriptions)} unique skill descriptions:")
    print("=" * 80)
    
    # Calculate total frames across all skills
    total_frames = sum(frame_counts.values())
    
    # Sort by frame count (descending) and print
    sorted_by_frames = sorted(frame_counts.items(), key=lambda x: x[1], reverse=True)
    
    for description, frames in sorted_by_frames:
        percentage = (frames / total_frames * 100) if total_frames > 0 else 0
        print(f"  {description:25s}  {frames:>12,} frames  ({percentage:5.2f}%)")
    
    print("=" * 80)
    print(f"Total: {len(unique_descriptions)} unique skill descriptions")
    print(f"Total frames: {total_frames:,}")

    print(f"Total episode lengths: {sum(episode_lengths):,}")
    print(f"Average episode length: {sum(episode_lengths) / len(episode_lengths):,.2f}")

#     # Save to file
#     output_file = "/root/openpi/unique_skill_descriptions.txt"
#     with open(output_file, 'w') as f:
#         f.write("Unique Skill Descriptions with Frame Counts\n")
#         f.write("=" * 80 + "\n\n")
#         for description, frames in sorted_by_frames:
#             percentage = (frames / total_frames * 100) if total_frames > 0 else 0
#             f.write(f"{description:25s}  {frames:>12,} frames  ({percentage:5.2f}%)\n")
#         f.write("\n" + "=" * 80 + "\n")
#         f.write(f"Total: {len(unique_descriptions)} unique skill descriptions\n")
#         f.write(f"Total frames: {total_frames:,}\n")
    
#     print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
