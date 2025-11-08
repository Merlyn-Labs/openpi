import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _is_distcp_checkpoint(checkpoint_dir: pathlib.Path | str) -> bool:
    """Check if the checkpoint directory contains distributed checkpoint files."""
    checkpoint_path = pathlib.Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return False

    # Check for .distcp files
    distcp_files = list(checkpoint_path.glob("*.distcp"))
    return len(distcp_files) > 0


def _load_distcp_checkpoint(checkpoint_dir: pathlib.Path | str, train_config, pytorch_device: str):
    """Load a distributed checkpoint and return the model."""
    try:
        import torch.distributed.checkpoint as dist_cp
        from torch.distributed.checkpoint import FileSystemReader
        import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
    except ImportError as e:
        raise ImportError(f"PyTorch and distributed checkpoint support required: {e}")

    checkpoint_path = pathlib.Path(checkpoint_dir)
    logging.info(f"Loading distributed checkpoint from {checkpoint_path} into {pytorch_device}")

    # Create model
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)

    # Load the distributed checkpoint
    state_dict = {
        "model": model.state_dict(),
    }

    # Use FileSystemReader to load the checkpoint
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=FileSystemReader(str(checkpoint_path)),
        no_dist=True,
    )

    # Load the state dict into the model
    model.load_state_dict(state_dict["model"])
    model.to(pytorch_device)

    logging.info("Successfully loaded distributed checkpoint")
    return model


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects the checkpoint format:
        - Distributed checkpoints (*.distcp files)
        - PyTorch safetensors checkpoints (model.safetensors)
        - JAX checkpoints (params directory)
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Determine the device to use for PyTorch models
    if pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # Check the checkpoint format
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_distcp = _is_distcp_checkpoint(checkpoint_dir)
    is_safetensors = os.path.exists(weight_path)
    is_pytorch = is_distcp or is_safetensors

    logging.info("Loading model...")
    if is_distcp:
        # Load from distributed checkpoint
        model = _load_distcp_checkpoint(checkpoint_dir, train_config, pytorch_device)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    elif is_safetensors:
        # Load from safetensors
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # Load from JAX checkpoint
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        # norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path("outputs/assets/pi05_b1k"), data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
