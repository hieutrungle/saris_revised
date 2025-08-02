import os
import warnings

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["BATCHED_PIPE_TIMEOUT"] = "1500"  # to avoid timeout errors in batched pipe
warnings.filterwarnings("ignore", category=UserWarning, module="torchrl")

from torchinfo import summary
from dataclasses import dataclass
import copy
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pyrallis
from typing import Callable
import torch.multiprocessing

torch.multiprocessing.set_start_method("forkserver", force=True)
from torchrl.envs import ParallelEnv, EnvBase, SerialEnv
import gc
from rs.utils import pytorch_utils, utils
from rs.envs import ENV_IDS
from rs.modules.agents import allocation, attention_critics, maac_critic

# Tensordict modules
from tensordict import TensorDict, from_module
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage, LazyMemmapStorage

# Env
from torchrl.envs import (
    Transform,
    Compose,
    RewardSum,
    TransformedEnv,
    StepCounter,
    ObservationNorm,
    DoubleToFloat,
)
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators
import torchrl.objectives.value

# Utils
torch.manual_seed(0)
from torchrl.record.loggers import TensorboardLogger, WandbLogger, Logger
from matplotlib import pyplot as plt
from tqdm import tqdm

set_composite_lp_aggregate(False).set()

from rs.run_ga import run_ga


@dataclass
class TrainConfig:

    # General arguments
    algo: str = "drl"  # the algorithm to run
    command: str = "train"  # the command to run
    load_model: str = "-1"  # Model load file name for resume training, "-1" doesn't load
    load_eval_model: str = "-1"  # Model load file name for evaluation, "-1" doesn't load
    load_allocator: str = "-1"  # Allocator load file name for resume training, "-1" doesn't load
    checkpoint_dir: str = "-1"  # the path to save the model
    allocator_replay_buffer_dir: str = "-1"  # the path to save the replay buffer
    load_allocator_replay_buffer: str = "-1"  # the path to load the replay buffer
    source_dir: str = "-1"  # the path to the source code
    verbose: bool = False  # whether to log to console
    seed: int = 10  # seed of the experiment
    eval_seed: int = 111  # seed of the evaluation
    save_interval: int = 100  # the interval to save the model
    start_step: int = 0  # the starting step of the experiment
    track_wandb: bool = True
    use_compile: bool = False  # whether to use torch.dynamo compiler
    image_dir: str = (
        "-1"  # the path to the image directory, if not provided, it will be set to source_dir
    )
    attention_dim: int = 128  # the dimension of the attention mechanism
    attention_heads: int = 4  # the number of attention heads
    start_idx: int = 0  # the starting index for the environment and allocator training
    drl_eval_results_dir: str = "-1"  # the path to save the DRL evaluation results

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 3  # the number of parallel environments
    ep_len: int = 50  # the maximum length of an episode
    eval_ep_len: int = 50  # the maximum length of an episode

    # Sampling
    frames_per_batch: int = 200  # Number of team frames collected per training iteration
    n_iters: int = 500  # Number of sampling and training iterations
    total_episodes: int = 3000  # Total number of episodes to run in the training

    # Training
    num_epochs: int = 40  # Number of optimization steps per training iteration
    minibatch_size: int = 200  # Size of the mini-batches in each optimization step
    lr: float = 2e-4  # Learning rate
    max_grad_norm: float = 1.0  # Maximum norm for the gradients

    # PPO
    clip_epsilon: float = 0.2  # clip value for PPO loss
    gamma: float = 0.985  # discount factor
    lmbda: float = 0.9  # lambda for generalised advantage estimation
    entropy_eps: float = 1e-4  # coefficient of the entropy term in the PPO loss

    # Wandb logging
    wandb_mode: str = "online"  # wandb mode
    project: str = "saris_revised"  # wandb project name
    group: str = "PPO_raw"  # wandb group name
    name: str = "FirstRun"  # wandb run name

    # Ablation Study
    no_compatibility_scores: bool = False  # whether to disable allocator compatibility scores
    random_assignment: bool = False  # whether to use random assignment of users to targets
    no_allocator: bool = False  # whether to disable the allocator

    def __post_init__(self):
        if self.source_dir == "-1":
            raise ValueError("Source dir is required for training")
        if self.checkpoint_dir == "-1":
            raise ValueError("Checkpoints dir is required for training")
        if self.algo.lower() == "ga" and self.drl_eval_results_dir == "-1":
            raise ValueError(
                "DRL evaluation results dir (--drl_eval_results_dir) is required for GA ALgorithm"
            )
        if self.sionna_config_file == "-1":
            raise ValueError("Sionna config file is required for training")
        if self.allocator_replay_buffer_dir == "-1":
            self.allocator_replay_buffer_dir = os.path.join(
                self.checkpoint_dir, "allocator_replay_buffer"
            )
        utils.mkdir_not_exists(self.checkpoint_dir)
        utils.mkdir_not_exists(self.allocator_replay_buffer_dir)

        self.frames_per_batch = self.frames_per_batch * self.num_envs
        # self.total_frames: int = self.frames_per_batch * self.n_iters
        self.allocator_path = os.path.join(self.checkpoint_dir, "allocator.pt")

        total_steps = self.total_episodes * self.ep_len
        n_iters = total_steps // self.frames_per_batch + 1
        self.n_iters = n_iters
        self.total_frames = self.frames_per_batch * self.n_iters

        device = pytorch_utils.init_gpu()
        self.device = device


class PolicyGradientAllocationLoss(nn.Module):
    """Policy gradient loss specifically for allocation tasks"""

    def __init__(self, entropy_coeff=0.01, value_coeff=0.5):
        super().__init__()
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

    def forward(self, tensordict_data):
        """Policy gradient loss with baseline
        Args:
            allocation_logits (torch.Tensor): Logits for allocation decisions, shape (batch_size, n_agents, n_targets)
            advantages (torch.Tensor): Advantages for the actions taken, shape (batch_size, n_agents)
            returns (torch.Tensor): Returns for the actions taken, shape (batch_size, n_agents)
            values (torch.Tensor): State values predicted by the critic, shape (batch_size, n_agents, 1)
        Returns:
            dict: A dictionary containing the total loss, policy loss, entropy, and value loss
        """
        values = tensordict_data.get(("allocator", "state_value"))
        advantages = tensordict_data.get(("next", "allocator", "advantage"))
        returns = tensordict_data.get(("next", "allocator", "returns"))

        new_allocation_logits = tensordict_data.get(("allocator", "new_allocation_logits"))
        location_target_dist = torch.distributions.Independent(
            base_distribution=torch.distributions.Categorical(logits=new_allocation_logits),
            reinterpreted_batch_ndims=1,
        )
        selected_loc_indices = tensordict_data.get(("allocator", "selected_loc_indices"))
        new_allocation_logprobs = location_target_dist.log_prob(selected_loc_indices)
        new_allocation_logprobs = new_allocation_logprobs.unsqueeze(-1)

        allocation_logprobs = tensordict_data.get(("allocator", "allocation_logprobs"))

        logratio = new_allocation_logprobs - allocation_logprobs
        ratio = logratio.exp()

        # Policy gradient loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Entropy bonus for exploration
        entropy = -location_target_dist.entropy().mean()

        # Value function loss
        value_loss = F.mse_loss(values, returns)

        # Total loss
        total_loss = policy_loss - self.entropy_coeff * entropy + self.value_coeff * value_loss

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "value_loss": value_loss,
        }


def get_allocator_loss(loss_module, allocator, allocator_target, tensordict_data):
    """Compute the loss for the allocator"""
    # Get the state values from the allocator
    allocator(tensordict_data)
    allocator_target(tensordict_data)

    # Get the rewards and dones
    rewards = tensordict_data.get(("next", "allocator", "reward"))
    # dones = tensordict_data.get(("next", "allocator", "done"))

    # Get the state values
    state_values = tensordict_data.get(("allocator", "state_value"))
    target_state_values = tensordict_data.get(("next", "allocator", "state_value"))

    # Compute advantages and returns
    advantages = rewards + 0.98 * target_state_values - state_values
    tensordict_data.set(("next", "allocator", "advantage"), advantages)
    returns = advantages + state_values
    tensordict_data.set(("next", "allocator", "returns"), returns)
    # Compute the loss
    loss_dict = loss_module(tensordict_data)
    return loss_dict


def wandb_init(config: TrainConfig) -> None:
    key_filename = os.path.join(config.source_dir, "tmp_wandb_api_key.txt")
    with open(key_filename, "r") as f:
        key_api = f.read().strip()
    wandb.login(relogin=True, key=key_api, host="https://api.wandb.ai")
    wandb.init(
        config=config,
        dir=config.checkpoint_dir,
        project=config.project,
        group=config.group,
        name=config.name,
        mode=config.wandb_mode,
    )


def make_env(config: TrainConfig, idx: int) -> Callable:

    def thunk() -> EnvBase:
        # Load Sionna configuration

        sionna_config = utils.load_config(config.sionna_config_file)
        sionna_config["seed"] = config.seed + idx
        sionna_config["num_runs_before_restart"] = 10
        scene_name = f"{sionna_config['scene_name']}_{idx}"
        sionna_config["scene_name"] = scene_name
        xml_dir = sionna_config["xml_dir"]
        xml_dir = os.path.join(xml_dir, scene_name)
        viz_scene_path = os.path.join(xml_dir, "idx", "scenee.xml")
        compute_scene_path = os.path.join(xml_dir, "ceiling_idx", "scenee.xml")
        sionna_config["xml_dir"] = xml_dir
        sionna_config["viz_scene_path"] = viz_scene_path
        sionna_config["compute_scene_path"] = compute_scene_path

        # image_dir = sionna_config["image_dir"]
        image_dir = os.path.join(config.image_dir, scene_name)
        # image_dir = config.image_dir
        sionna_config["image_dir"] = image_dir

        if config.command.lower() == "train":
            sionna_config["rendering"] = False
        else:
            sionna_config["rendering"] = True

        env_kwargs = {
            "sionna_config": sionna_config,
            "allocator_path": config.allocator_path,
            "seed": config.seed,
            "device": config.device,
            "num_runs_before_restart": 20,
            "random_assignment": config.random_assignment,
            "no_allocator": config.no_allocator,
            "no_compatibility_scores": config.no_compatibility_scores,
        }

        if config.algo.lower() == "ga":
            results = torch.load(config.drl_eval_results_dir, weights_only=False)
            target_pos = results["agents", "target_pos"]
            rx_positions = target_pos[0, 0, :, :3, :]
            rx_positions = rx_positions[::20, ...]
            env_kwargs["rx_positions"] = rx_positions

        if config.command.lower() == "eval":
            env_kwargs["eval_mode"] = True
            env_kwargs["seed"] = config.eval_seed

        if config.env_id.lower() not in ENV_IDS:
            raise ValueError(f"Unknown environment id: {config.env_id}")
        env_cls = ENV_IDS[config.env_id.lower()]
        env = env_cls(**env_kwargs)

        return env

    return thunk


@pyrallis.wrap()
def main(config: TrainConfig):

    if config.command.lower() == "train":
        print(f"=" * 30 + "Training" + "=" * 30)
    else:
        print(f"=" * 30 + "Evaluation" + "=" * 30)

    utils.log_config(config.__dict__)

    # Reset the torch compiler if needed
    torch.compiler.reset()
    torch.multiprocessing.set_start_method("forkserver", force=True)
    pytorch_utils.init_seed(config.seed)

    # envs = SerialEnv(config.num_envs, [make_env(config, idx) for idx in range(config.num_envs)])
    # check_env_specs(envs)

    envs = ParallelEnv(
        config.num_envs,
        [make_env(config, idx) for idx in range(config.num_envs)],
        mp_start_method="forkserver",
        shared_memory=False,
    )
    ob_spec = envs.observation_spec
    ac_spec = envs.action_spec

    observation_shape = ob_spec["agents", "observation"].shape
    loc = torch.zeros(observation_shape, device=config.device)
    scale = torch.ones(observation_shape, device=config.device) * 8.0

    if config.algo.lower() == "ga":
        compose = Compose(
            ObservationNorm(
                loc=loc,
                scale=scale,
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "observation")],
                standard_normal=True,
            ),
            DoubleToFloat(),
            RewardSum(in_keys=[("agents", "reward")], out_keys=[("agents", "episode_reward")]),
        )
    else:
        compose = Compose(
            ObservationNorm(
                loc=loc,
                scale=scale,
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "observation")],
                standard_normal=True,
            ),
            DoubleToFloat(),
            StepCounter(max_steps=config.ep_len),
            RewardSum(in_keys=[("agents", "reward")], out_keys=[("agents", "episode_reward")]),
        )

    envs = TransformedEnv(envs, compose)

    if loc is None and scale is None:
        envs.transform[0].init_stats(num_iter=config.ep_len * 3, reduce_dim=(0, 1, 2), cat_dim=1)

    if config.algo.lower() == "drl":
        run_drl(envs, config)
    elif config.algo.lower() == "ga":
        run_ga(envs, config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algo}. Supported: 'drl', 'ga'.")


def run_drl(envs: ParallelEnv, config: TrainConfig):
    try:
        if envs.is_closed:
            envs.start()

        ob_spec = envs.observation_spec
        ac_spec = envs.action_spec

        checkpoint = None
        if config.load_model != "-1":
            checkpoint = torch.load(config.load_model)
            print(f"Loaded checkpoint from {config.load_model}")

        n_agents = list(envs.n_agents)[0]
        shared_parameters_policy = False
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                # n_obs_per_agent
                n_agent_inputs=ob_spec["agents", "observation"].shape[-1],
                n_agent_outputs=2 * ac_spec.shape[-1],  # 2 * n_actions_per_agents
                n_agents=n_agents,
                centralised=False,
                share_params=shared_parameters_policy,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.ReLU6,
            ),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=envs.action_spec_unbatched,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[envs.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": envs.full_action_spec_unbatched[envs.action_key].space.low,
                "high": envs.full_action_spec_unbatched[envs.action_key].space.high,
            },
            return_log_prob=True,
        )  # we'll need the log-prob for the PPO loss

        share_parameters_critic = False
        mappo = True
        critic_net = MultiAgentMLP(
            n_agent_inputs=ob_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,  # 1 value per agent
            n_agents=n_agents,
            centralised=mappo,
            share_params=share_parameters_critic,
            device=config.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        critic = TensorDictModule(
            module=critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )

        collector = SyncDataCollector(
            envs,
            policy,
            device=config.device,
            storing_device=config.device,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.total_frames,
        )

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(config.frames_per_batch, device=config.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=config.minibatch_size,
        )

        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            loss_critic_type="l2",
            normalize_advantage=True,
            normalize_advantage_exclude_dims=(1,),
            clip_epsilon=config.clip_epsilon,
            entropy_bonus=config.entropy_eps > 0,
            entropy_coef=config.entropy_eps,
            # Important to avoid normalizing across the agent dimension
            # normalize_advantage=False,
        )
        loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=("agents", "reward"),
            action=envs.action_key,
            value=("agents", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        # GAE
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda
        )

        # optimizer
        optim = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

        if checkpoint:
            print(f"Loading checkpoint from: {config.load_model}")
            policy.load_state_dict(checkpoint["policy"])
            if config.command.lower() == "train":
                critic.load_state_dict(checkpoint["critic"])
                loss_module.load_state_dict(checkpoint["loss_module"])
                optim.load_state_dict(checkpoint["optimizer"])

        if config.command == "train":
            print("Training...")
            train(
                envs=envs,
                config=config,
                collector=collector,
                policy=policy,
                critic=critic,
                loss_module=loss_module,
                optim=optim,
                replay_buffer=replay_buffer,
            )
        else:
            print("Evaluation...")
            eval(envs, config, policy)
            print("Evaluation done")

    except Exception as e:
        print("Environment specs are not correct")
        print(e)
        traceback.print_exc()
    finally:
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()
        if not envs.is_closed:
            envs.close()


def train(
    envs: ParallelEnv,
    config: TrainConfig,
    collector: SyncDataCollector,
    policy: TensorDictModule,
    critic: TensorDictModule,
    loss_module: ClipPPOLoss,
    optim: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
):

    if config.track_wandb:
        wandb_init(config)
    else:
        logger = TensorboardLogger(
            exp_name=f"{config.group}_{config.name}",
            log_dir=config.checkpoint_dir,
        )
        saved_config = config.__dict__.copy()
        saved_config["device"] = str(config.device)
        logger.log_hparams(saved_config)

    # allocator_loss_module = PolicyGradientAllocationLoss(
    #     entropy_coeff=config.entropy_eps, value_coeff=0.5
    # )

    pbar_iterable = range(config.start_idx, config.n_iters)
    pbar = tqdm(
        pbar_iterable,
        total=config.n_iters,
        desc="episode_reward_mean = 0.0",
        initial=config.start_idx,
    )
    GAE = loss_module.value_estimator
    episode_reward_mean_list = []
    for idx, tensordict_data in enumerate(collector, start=config.start_idx):

        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
        # Agents
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", ("agents", "reward")))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", ("agents", "reward")))),
        )
        tensordict_data.set(
            ("next", "agents", "truncated"),
            tensordict_data.get(("next", "truncated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", ("agents", "reward")))),
        )

        with torch.no_grad():
            # Compute GAE and add it to the data
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)

        for i in range(config.num_epochs):
            for _ in range(config.frames_per_batch // config.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config.max_grad_norm)
                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        )
        done = tensordict_data.get(("next", "done"))
        episode_max_step = tensordict_data.get(("next", "step_count"))[done]
        episode_max_step = episode_max_step.to(torch.float).mean().item()
        episode_reward_mean_list.append(episode_reward_mean)
        logs = {
            "episode_max_step": episode_max_step,
            "episode_reward_mean": episode_reward_mean,
            "loss_objective": loss_vals["loss_objective"].item(),
            "loss_critic": loss_vals["loss_critic"].item(),
            "loss_entropy": loss_vals["loss_entropy"].item(),
        }
        step = idx * config.frames_per_batch
        if config.track_wandb:
            wandb.log({**logs}, step=step)
        else:
            for key, value in logs.items():
                logger.log_scalar(key, value, step=step)
        torch.save(
            {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "loss_module": loss_module.state_dict(),
                "optimizer": optim.state_dict(),
            },
            os.path.join(config.checkpoint_dir, f"checkpoint_{idx}.pt"),
        )
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
        pbar.update()


def eval(envs: ParallelEnv, config: TrainConfig, policy: TensorDictModule):
    with torch.no_grad():
        collector = SyncDataCollector(
            envs,
            policy,
            device=config.device,
            storing_device=config.device,
            frames_per_batch=config.ep_len * 20 * config.num_envs,
            total_frames=config.ep_len * 20 * config.num_envs,
        )
        for idx, tensordict_data in enumerate(collector):
            rollouts = tensordict_data

    # save the rollout
    image_dir = config.image_dir
    # get last name of the image directory
    name = image_dir.split("/")[-1]
    rollout_path = os.path.join(config.checkpoint_dir, f"env_rollout_{name}.pt")
    torch.save(rollouts, rollout_path)
    print(f"Rollout saved to {rollout_path}")


def get_allocator_tensordict(tensordict_data: TensorDict) -> TensorDict:
    """Get the tensordict for the allocator"""
    # select data based on done signal
    if tensordict_data.get(("next", "done")) is None:
        raise ValueError("The tensordict does not contain the 'done' key.")
    step_count = tensordict_data.get(("step_count"))
    dones = tensordict_data.get(("next", "done"))
    keys = [
        ("allocator", "agent_states"),
        ("allocator", "init_agent_states"),
        ("allocator", "target_states"),
        ("allocator", "compatibility"),
        ("allocator", "allocation_logits"),
        ("allocator", "allocation_logprobs"),
        ("allocator", "selected_loc_indices"),
        ("next", "allocator", "episode_reward"),
        ("next", "allocator", "done"),
        ("next", "allocator", "terminated"),
        ("next", "allocator", "truncated"),
    ]
    new_keys = [
        ("next", "allocator", "agent_states"),
        ("allocator", "agent_states"),
        ("allocator", "target_states"),
        ("allocator", "compatibility"),
        ("allocator", "allocation_logits"),
        ("allocator", "allocation_logprobs"),
        ("allocator", "selected_loc_indices"),
        ("next", "allocator", "reward"),
        ("next", "allocator", "done"),
        ("next", "allocator", "terminated"),
        ("next", "allocator", "truncated"),
    ]
    tmp_allocator_tensordict = tensordict_data.select(*keys)
    # rename the keys
    for key, new_key in zip(keys, new_keys):
        tmp_allocator_tensordict.rename_key_(key, new_key)

    allocator_tensordict = {}
    for key in new_keys:
        # get values if done = True
        vals = []
        for i, done in enumerate(dones):
            if done:
                vals.append(tmp_allocator_tensordict.get(key)[i])
        allocator_tensordict[key] = torch.stack(vals, dim=0)
        if "reward" in key:
            allocator_tensordict[key] /= step_count[dones].to(torch.float).unsqueeze(-1)

    allocator_tensordict = TensorDict(
        allocator_tensordict,
        batch_size=[len(allocator_tensordict["next", "allocator", "reward"])],
        device=tensordict_data.device,
    )

    allocator_tensordict.set(
        ("next", "allocator", "target_states"),
        allocator_tensordict.get(("allocator", "target_states")),
    )
    allocator_tensordict.set(
        ("next", "allocator", "compatibility"),
        allocator_tensordict.get(("allocator", "compatibility")),
    )
    return allocator_tensordict


if __name__ == "__main__":
    main()
