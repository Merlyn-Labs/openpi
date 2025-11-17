import dataclasses
import enum
import logging
import socket

import tyro

from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Dataset root, used to retrieve the prompt of the task if taskname is not None.
    dataset_root: str | None = "/scr/behavior/2025-challenge-demos"
    # If provided, will be used to retrieve the prompt of the task, otherwise use turning_on_radio as default.
    task_name: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Control mode
    control_mode: str = "receeding_temporal"
    max_len: int = 72
    action_horizon: int = 12
    temporal_ensemble_max: int = 6
    exp_k_value: float = 1.0


def create_policy(args: Args, config: _config.TrainConfig) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        config, args.policy.dir, default_prompt=args.default_prompt
    )


def main(args: Args) -> None:
    config = _config.get_config(args.policy.config)
    policy = create_policy(args, config)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    policy = B1KPolicyWrapper(policy, config=config, text_prompt="Hello, world!", control_mode=args.control_mode, max_len=args.max_len, action_horizon=args.action_horizon, temporal_ensemble_max=args.temporal_ensemble_max, exp_k_value=args.exp_k_value)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
