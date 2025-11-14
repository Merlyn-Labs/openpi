import subprocess
import shutil
from pathlib import Path
import time
import os

ONE_MINUTE = 60

idx = 0
all_checkpoints_dir = Path("outputs/checkpoints")

while True:
    config_names = os.listdir(all_checkpoints_dir)
    idx += 1
    print(f"Running iteration {idx}...")

    for config_name in config_names:
        config_specific_checkpoints_dir = all_checkpoints_dir / config_name
        experiment_names = os.listdir(config_specific_checkpoints_dir)
        print(f"Processing config {config_name} with {len(experiment_names)} experiments...")

        for experiment_name in experiment_names:
            base_local = config_specific_checkpoints_dir / experiment_name

            # Determine steps dynamically by listing subdirectories in base_local that are integers
            steps_to_sync = sorted(
                int(p.name)
                for p in base_local.iterdir()
                if p.is_dir() and p.name.isdigit()
            )
            if len(steps_to_sync) == 0:
                continue
            steps_to_delete = steps_to_sync[:-1]
            print(f"steps_to_sync={steps_to_sync}, steps_to_delete={steps_to_delete}")

            for step in steps_to_sync:
                print(f"Syncing {step}")
                local_path = base_local / f"{step}"
                s3_path = f"s3://behavior-challenge/{os.path.normpath(str(local_path))}"
                successfully_synced = False
                try:
                    subprocess.run(
                        ["aws", "s3", "sync", local_path, s3_path],
                        check=True
                    )
                    successfully_synced = True
                except subprocess.CalledProcessError:
                    pass

                # After successful sync, remove the local directory if 2 or more checkpoints are present
                if step in steps_to_delete and successfully_synced:
                    shutil.rmtree(Path(local_path), ignore_errors=True)

    time.sleep(ONE_MINUTE)
