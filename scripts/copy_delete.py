import subprocess
import shutil
from pathlib import Path
import time

ONE_HOUR = 3600
base_local = "outputs/checkpoints/pi0_b1k/openpi_20250929_035039/".strip()
base_s3 = f"s3://behavior-challenge/{base_local}"
idx = 0

while True:
    idx += 1

    # Determine steps dynamically by listing subdirectories in base_local that are integers
    steps_to_sync = sorted(
        int(p.name)
        for p in Path(base_local).iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    steps_to_delete = steps_to_sync[:-1]
    print(f"Iteration {idx}, syncing steps {steps_to_sync}")

    print(f"steps_to_sync={steps_to_sync}, steps_to_delete={steps_to_delete}")

    for step in steps_to_sync:
        print(f"Syncing {step}")
        local_path = f"{base_local}/{step}/"
        s3_path = f"{base_s3}/{step}/"
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

    time.sleep(ONE_HOUR)
