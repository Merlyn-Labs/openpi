import json
import os
from concurrent.futures import ThreadPoolExecutor
import random
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Base directories (without task-xxxx)
annotations_base_dir = "/scratch/vision/group/behavior/2025-challenge-demos/annotations/"
# annotations_base_dir = "/root/.cache/huggingface/hub/datasets--behavior-1k--2025-challenge-demos/snapshots/48908ec3c6e64cbb696d7cbfa023568be2a8abcf/annotations/"
prompt_base_dir = annotations_base_dir.replace("annotations/", "skill_prompts/")

action_keys_to_ignore = ["skill_idx", "skill_id", "frame_duration", "mp_ef"]
existing_prompts = []

client = OpenAI()
model = "gpt-5"

def preprocess_actions(actions):
    """Preprocess the actions to remove the skill_idx, skill_id, and mp_ef fields."""
    return [
        {k: v for k, v in action.items() if k not in action_keys_to_ignore}
        for action in actions
    ]

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def generate_prompt_from_action(task_name, actions, existing_prompts):
    """Generate a prompt from an action JSON via gpt-5-nano, with retry logic."""
    system_prompt = """You are an expert in prompt engineering for robotics and Vision-Language-Action (VLA) models. Your task is to convert a structured JSON skill annotation from the Stanford BEHAVIOR dataset into a high-quality, concise natural-language instruction suitable for fine-tuning a Vision-Language-Action (VLA) model.

The JSON represents a single robotic skill and typically includes:

"skill_description": A list with the main action phrase, e.g., ["move to"] or ["pick up from"].
"object_id": A list of involved objects or containers, e.g., [["radio_89"]] or [["radio_89", "coffee_table_koagbh_0"]]. Use the object names naturally, omitting numeric suffixes like "_89".
"manipulating_object_id": The primary object being handled (if present, emphasize it).
"memory_prefix": Modifiers like ["back"] (incorporate if relevant, e.g., "place back on").
"spatial_prefix": Spatial details (incorporate if relevant, e.g., "inside").
"skill_type": Category like ["navigation"] or ["uncoordinated"] (use to inform tone, e.g., simple for navigation).

Guidelines for the generated instruction:

- Make it concise (10-15 words max), natural, and robot-executable (start with an imperative verb).
- Focus on the core action, objects, and any modifiers.
- Use everyday language for generalization (e.g., "the radio" instead of "radio_89").
- Ensure it's suitable as a VLA prompt: Descriptive enough for vision-guided action but not verbose.
- If fields are empty or irrelevant, omit them.
- Important: don't always use the same words for the same action. Add some variation to the instructions. Refer to the existing prompts input to avoid repeating those existing instructions that have been recently generated.
- Output ONLY the generated instructions as a JSON array of strings.

Here are some examples:

Example 1
Task name: tidying bedroom
Skill annotation JSON array: [{'skill_description': ['move to'], 'object_id': [['sandal_190']], 'manipulating_object_id': [], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['pick up from'], 'object_id': [['sandal_190', 'floors_htolat_0']], 'manipulating_object_id': ['sandal_190'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}, {'skill_description': ['move to'], 'object_id': [['sandal_189']], 'manipulating_object_id': [], 'memory_prefix': ['the other'], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['pick up from'], 'object_id': [['sandal_189', 'floors_htolat_0']], 'manipulating_object_id': ['sandal_189'], 'memory_prefix': ['the other'], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}, {'skill_description': ['move to'], 'object_id': [['bed_gxfipj_0']], 'manipulating_object_id': [], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['place on next to'], 'object_id': [['sandal_190', 'floors_htolat_0', 'bed_gxfipj_0']], 'manipulating_object_id': ['sandal_190'], 'memory_prefix': [], 'spatial_prefix': [['', '', 'left']], 'skill_type': ['uncoordinated']}, {'skill_description': ['place on next to'], 'object_id': [['sandal_189', 'floors_htolat_0', 'bed_gxfipj_0']], 'manipulating_object_id': ['sandal_189'], 'memory_prefix': [], 'spatial_prefix': [['', '', 'left']], 'skill_type': ['uncoordinated']}, {'skill_description': ['move to'], 'object_id': [['hardback_188']], 'manipulating_object_id': [], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['push to'], 'object_id': [['hardback_188', 'bed_gxfipj_0']], 'manipulating_object_id': ['hardback_188'], 'memory_prefix': [], 'spatial_prefix': [['', 'to_the_edge_of']], 'skill_type': ['uncoordinated']}, {'skill_description': ['pick up from'], 'object_id': [['hardback_188', 'bed_gxfipj_0']], 'manipulating_object_id': ['hardback_188'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}, {'skill_description': ['move to'], 'object_id': [['nightstand_wbxekb_0']], 'manipulating_object_id': [], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['place on'], 'object_id': [['hardback_188', 'nightstand_wbxekb_0']], 'manipulating_object_id': ['hardback_188'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}]
Existing prompts: [
"Navigate to the sandal.",
"Pick up the sandal from the floor.",
"Go to the other sandal.",
"Grab the other sandal from the floor.",
"Head to the bed.",
"Set the sandal on the floor to the left of the bed.",
"Position the other sandal on the floor left of the bed.",
"Approach the hardback book.",
"Nudge the hardback to the edge of the bed.",
"Lift the hardback from the bed.",
"Proceed to the nightstand.",
"Put the hardback on the nightstand."
]

Output: [
"Approach the shoe.",
"Grasp the sandal off the ground.",
"Head toward the matching shoe.",
"Retrieve the matching shoe from the ground.",
"Proceed toward the bed.",
"Position the shoe on the ground next to the bed on the left.",
"Set the matching shoe on the ground beside the bed on the left.",
"Go toward the book.",
"Slide the book toward the bed's edge.",
"Take the book off the bed.",
"Advance to the nightstand.",
"Deposit the book onto the nightstand."
]

Example 2
Task name: picking up trash
Skill annotation JSON array: [{'skill_description': ['move to'], 'object_id': [['trash_can_116']], 'manipulating_object_id': [], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['pick up from'], 'object_id': [['trash_can_116', 'floors_zqjkvm_0']], 'manipulating_object_id': ['trash_can_116'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}, {'skill_description': ['move to'], 'object_id': [['can_of_soda_114']], 'manipulating_object_id': [], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['pick up from'], 'object_id': [['can_of_soda_114', 'floors_ulujpr_0']], 'manipulating_object_id': ['can_of_soda_114'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}, {'skill_description': ['place in'], 'object_id': [['can_of_soda_114', 'trash_can_116']], 'manipulating_object_id': ['can_of_soda_114'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['coordinated']}, {'skill_description': ['move to'], 'object_id': [['can_of_soda_115']], 'manipulating_object_id': [], 'memory_prefix': ['the other'], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['pick up from'], 'object_id': [['can_of_soda_115', 'floors_ulujpr_0']], 'manipulating_object_id': ['can_of_soda_115'], 'memory_prefix': ['the other'], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}, {'skill_description': ['place in'], 'object_id': [['can_of_soda_115', 'trash_can_116']], 'manipulating_object_id': ['can_of_soda_115'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['coordinated']}, {'skill_description': ['move to'], 'object_id': [['can_of_soda_113']], 'manipulating_object_id': [], 'memory_prefix': ['the other'], 'spatial_prefix': [], 'skill_type': ['navigation']}, {'skill_description': ['pick up from'], 'object_id': [['can_of_soda_113', 'floors_ulujpr_0']], 'manipulating_object_id': ['can_of_soda_113'], 'memory_prefix': ['the other'], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}, {'skill_description': ['place in'], 'object_id': [['can_of_soda_113', 'trash_can_116']], 'manipulating_object_id': ['can_of_soda_113'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['coordinated']}, {'skill_description': ['place on'], 'object_id': [['trash_can_116', 'floors_ulujpr_0']], 'manipulating_object_id': ['trash_can_116'], 'memory_prefix': [], 'spatial_prefix': [], 'skill_type': ['uncoordinated']}]
Existing prompts: [
"Go to the trash can.",
"Pick up the trash can from the floor.",
"Head to the soda can.",
"Grab the soda can from the floor.",
"Drop the soda can into the trash can.",
"Navigate to the other soda can.",
"Lift the other soda can from the floor.",
"Insert the other soda can in the trash can.",
"Approach the remaining soda can.",
"Retrieve the remaining soda can from the floor.",
"Put the remaining soda can into the trash can.",
"Set the trash can on the floor."
]

Output: [
"Approach the bin.",
"Grasp the bin off the ground.",
"Proceed toward the drink can.",
"Take the drink can off the ground.",
"Deposit the drink can inside the bin.",
"Head toward the additional drink can.",
"Collect the additional drink can from the ground.",
"Toss the additional drink can into the bin.",
"Advance to the last drink can.",
"Snatch the last drink can off the ground.",
"Place the last drink can in the bin.",
"Position the bin onto the ground."
]

Now, generate the instruction for the provided input JSON."""
    preprocessed_actions = preprocess_actions(actions)
    user_content = f"Task name: {task_name}\nSkill annotation JSON array:\n{json.dumps(preprocessed_actions, indent=2)}\nExisting prompts:\n{json.dumps(existing_prompts, indent=2)}"
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_content,
    )
    generated_prompts = json.loads(response.output_text)
    assert len(generated_prompts) == len(actions), f"Generated {len(generated_prompts)} prompts for {task_name} but expected {len(actions)}, response: {response.output_text}"
    return generated_prompts

def process_file(annotations_dir, prompt_out_dir, file, idx, total_files, existing_prompts):
    out_file_path = os.path.join(prompt_out_dir, file)

    if os.path.exists(out_file_path):
        print(f"Skipping {file} because it already exists ({idx + 1}/{total_files})")
        return

    with open(os.path.join(annotations_dir, file), "r") as f:
        data = json.load(f)

    task_name = data["task_name"]
    actions = data["skill_annotation"]

    prompts_list = []
    try:
        prompts = generate_prompt_from_action(task_name, actions, random.sample(existing_prompts, k=min(15, len(existing_prompts))))
        for prompt, action in zip(prompts, actions):
            prompts_list.append({
                "prompt": prompt,
                "frame_duration": action["frame_duration"],
            })
            existing_prompts.append(prompt)
    except Exception as e:
        print(f"Error generating prompts for {file}: {e}")
        return

    with open(out_file_path, "w") as f:
        json.dump(prompts_list, f, indent=2)
        print(f"Wrote prompts to {out_file_path} ({idx + 1}/{total_files})")

# Find all task-xxxx directories in the annotations base dir
task_dirs = sorted(
    [d for d in os.listdir(annotations_base_dir) if re.match(r"task-\d{4}", d)],
    reverse=True
)

for task_dir in task_dirs:
    annotations_dir = os.path.join(annotations_base_dir, task_dir)
    prompt_out_dir = os.path.join(prompt_base_dir, task_dir)
    os.makedirs(prompt_out_dir, exist_ok=True)

    files = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
    total_files = len(files)

    print(f"Processing {total_files} files in {annotations_dir}...")

    with ThreadPoolExecutor(max_workers=32) as executor:
        for idx, file in enumerate(files):
            executor.submit(process_file, annotations_dir, prompt_out_dir, file, idx, total_files, existing_prompts)
