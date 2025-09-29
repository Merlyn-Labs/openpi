GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e 

source .venv/bin/activate

# Install behavior for server deploy 
cd ~/BEHAVIOR-1K/
uv pip install -e bddl
uv pip install -e OmniGibson[eval]
