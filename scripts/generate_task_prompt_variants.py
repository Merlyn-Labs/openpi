import json
import os
from concurrent.futures import ThreadPoolExecutor
import random

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configuration
episodes_file = "/scratch/vision/group/behavior/2025-challenge-demos/meta/episodes.jsonl"
output_file = "/scratch/vision/group/behavior/2025-challenge-demos/meta/episodes_with_variants.jsonl"
progress_file = "/scratch/vision/group/behavior/2025-challenge-demos/meta/prompt_variants_progress.json"

client = OpenAI()
model = "gpt-5"

EPISODES_PER_TASK = 200
MAX_WORKERS = 4

ACCEPTABLE_TASKS_LIST = [0,1,3,4,5,7,13,15,16,17,18,20,22,23,25,27,30,33,34,35,37,38,39,40,42,43,45,46,47,48,49]

def load_progress():
    """Load progress from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Save progress to the progress file."""
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def generate_prompt_variants(original_prompt, num_variants):
    """Generate unique variants of a prompt using GPT-5-nano, with retry logic."""
    system_prompt = """You are an expert in prompt engineering for robotics and Vision-Language-Action (VLA) models. Your task is to generate unique variants of a given robot instruction prompt.

Guidelines for generating variants:

1. **Novel Phrasing**: Use different nouns, verbs, adverbs, and adjectives to create variety.
   - Example: "Pick up" → "Grasp", "Retrieve", "Take", "Collect", "Grab"
   - Example: "Place" → "Put", "Set", "Position", "Deposit", "Arrange"
   - Example: "the radio" → "the radio receiver", "the radio device"

2. **Simplification is OK**: You can remove details to make prompts higher-level.
   - Example: "Pick up the radio from the table and turn it on." → "Grasp the radio and turn it on."
   - Example: "Turn on the radio receiver that's on the table in the living room." → "Turn on the radio in the living room."

3. **NEVER Fabricate Details**: Do not add incorrect or new information that wasn't in the original prompt.
   - BAD: Adding "gently" if it wasn't in the original
   - BAD: Adding specific locations or objects not mentioned
   - GOOD: Removing or rephrasing existing details

4. **Maintain Core Intent**: The task's goal must remain the same.

5. **Keep It Concise**: Robot instructions should be clear and actionable (typically 10-20 words).

6. **Output Format**: Return ONLY a JSON array of strings, each string being one variant.

Examples:

Original: "Turn on the radio receiver that's on the table in the living room."
Variants:
[
"Activate the radio on the living room table.",
"Switch on the radio receiver in the living room.",
"Turn the radio on in the living room.",
"Power on the table radio in the living room.",
"Enable the radio device on the table.",
"Turn on the living room radio.",
"Activate the radio receiver from the table.",
"Switch the living room radio on."
]

Original: "Put the three can of soda from the living room inside the tash can in the kitchen."
Variants:
[
"Place the three soda cans from the living room into the kitchen trash can.",
"Move all three sodas to the kitchen trash bin.",
"Transfer the three living room soda cans to the kitchen trash.",
"Dispose of the three soda cans in the kitchen trash.",
"Take the three sodas from the living room and put them in the kitchen bin.",
"Collect the three soda cans and place them in the kitchen trash.",
"Bring the three living room sodas to the kitchen trash can."
]

Now generate the variants for the provided prompt."""

    user_content = f"Original prompt: {original_prompt}\nNumber of variants needed: {num_variants}"
    
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_content,
    )
    
    generated_variants = json.loads(response.output_text)
    
    # Validate we got the right number of variants
    if len(generated_variants) != num_variants:
        print(f"Warning: Generated {len(generated_variants)} variants but requested {num_variants}")
    
    return generated_variants


def load_episodes():
    """Load all episodes from the episodes.jsonl file."""
    episodes = []
    with open(episodes_file, "r") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def group_episodes_by_task(episodes):
    """Group episodes by their task prompt."""
    task_to_episodes = {}
    for episode in episodes:
        task = episode["tasks"][0]
        if task not in task_to_episodes:
            task_to_episodes[task] = []
        task_to_episodes[task].append(episode)
    return task_to_episodes


def process_task(task_index, task_prompt, episodes, progress):
    """Generate variants for a single task and return updated episodes."""
    task_id = task_prompt[:50]  # Use first 50 chars as identifier
    
    # Check if we've already processed this task
    if task_id in progress and task_index in ACCEPTABLE_TASKS_LIST:  # we wanna do all now, so explicitly do the ones which were previously not in acceptable tasks list
        print(f"Skipping task (already processed): {task_prompt[:80]}...")
        return None
    
    print(f"Processing task {task_index}: {task_id}...")

    # if task_index not in ACCEPTABLE_TASKS_LIST:
    #     print(f"Skipping task {task_index} (not in acceptable tasks list): {task_id}...")
    #     return task_id, episodes
    # else:
    #     print(f"Processing task {task_index}: {task_id}...")

    try:
        # Generate 199 new variants (we already have 1 original)
        num_variants_needed = len(episodes) - 1
        print(f"Generating {num_variants_needed} variants for task: {task_prompt[:80]}...")
        
        variants = generate_prompt_variants(task_prompt, num_variants_needed)
        
        # Keep the original for the first episode, assign variants to the rest
        all_prompts = [task_prompt] + variants
        
        # Shuffle variants to avoid any bias in assignment
        variant_pool = all_prompts[1:]  # Exclude original
        random.shuffle(variant_pool)
        
        # Assign prompts to episodes
        updated_episodes = []
        for i, episode in enumerate(episodes):
            updated_episode = episode.copy()
            if i == 0:
                # Keep original for first episode
                updated_episode["tasks"] = [task_prompt]
            else:
                # Assign a variant
                updated_episode["tasks"] = [variant_pool[i - 1]]
            updated_episodes.append(updated_episode)
        
        print(f"✓ Completed task: {task_prompt[:80]}...")
        return (task_id, updated_episodes)
        
    except Exception as e:
        print(f"✗ Error processing task '{task_prompt[:80]}...': {e}")
        return None


def main():
    print("Loading episodes...")
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes")
    
    print("Grouping episodes by task...")
    task_to_episodes = group_episodes_by_task(episodes)
    print(f"Found {len(task_to_episodes)} unique tasks")
    
    # Verify each task has 200 episodes
    for task, eps in task_to_episodes.items():
        if len(eps) != EPISODES_PER_TASK:
            print(f"Warning: Task has {len(eps)} episodes (expected {EPISODES_PER_TASK}): {task[:80]}...")
    
    # Load progress
    progress = load_progress()
    print(f"Progress: {len(progress)}/{len(task_to_episodes)} tasks completed")
    
    # Create a dictionary to store all updated episodes by episode_index
    all_updated_episodes = {}
    
    # Load any previously completed work
    if os.path.exists(output_file):
        print("Loading previously saved episodes...")
        with open(output_file, "r") as f:
            for line in f:
                episode = json.loads(line)
                all_updated_episodes[episode["episode_index"]] = episode
    
    # Process tasks
    tasks_list = list(task_to_episodes.items())
    total_tasks = len(tasks_list)
    
    for task_index, (task_prompt, episodes) in enumerate(tasks_list):
        task_id = task_prompt[:50]
        
        if task_id in progress and task_index in ACCEPTABLE_TASKS_LIST:  # we wanna do all now, so explicitly do the ones which were previously not in acceptable tasks list
            continue
        
        result = process_task(task_index, task_prompt, episodes, progress)
        
        if result is not None:
            task_id, updated_episodes = result
            
            # Update the all_updated_episodes dictionary
            for episode in updated_episodes:
                all_updated_episodes[episode["episode_index"]] = episode
            
            # Mark as complete
            progress[task_id] = True
            save_progress(progress)
            
            # Write all episodes to output file (sorted by episode_index)
            with open(output_file, "w") as f:
                for episode_index in sorted(all_updated_episodes.keys()):
                    f.write(json.dumps(all_updated_episodes[episode_index]) + "\n")
            
            print(f"Progress: {len(progress)}/{total_tasks} tasks completed\n")
    
    print("\n" + "="*80)
    print("All tasks completed!")
    print(f"Output written to: {output_file}")
    print(f"Total episodes processed: {len(all_updated_episodes)}")
    print("="*80)


if __name__ == "__main__":
    main()

