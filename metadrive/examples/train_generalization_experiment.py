"""
This script demonstrates how to train a set of policies under different number of training scenarios and test them
in the same test set using rllib.

We verified this script with ray==2.2.0. Please report to use if you find newer version of ray is not compatible with
this script. Installation guide:

    pip install ray[rllib]==2.2.0
    pip install tensorflow_probability==0.24.0
    pip install torch

"""
import argparse
import copy
import logging
from typing import Dict

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.envs.gym_wrapper import createGymWrapper, GymToGymnasiumWrapper

try:
    import ray
    from ray import tune
    from ray.tune.registry import register_env

    from ray.tune import CLIReporter
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.env import BaseEnv
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
    from ray.rllib.evaluation import RolloutWorker
    from ray.rllib.policy import Policy
except ImportError:
    ray = None
    raise ValueError("Please install ray through 'pip install ray'.")


class DrivingCallbacks(DefaultCallbacks):
    # ---------- Collecte pendant le rollout ----------
    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
        episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        episode.user_data.update(
            velocity=[], steering=[], step_reward=[], acceleration=[], cost=[]
        )

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode,
        env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info:
            ud = episode.user_data
            ud["velocity"].append(info["velocity"])
            ud["steering"].append(info["steering"])
            ud["step_reward"].append(info["step_reward"])
            ud["acceleration"].append(info["acceleration"])
            ud["cost"].append(info["cost"])

    def on_episode_end(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
        episode: MultiAgentEpisode, **kwargs
    ):
        info = episode.last_info_for()
        arrive_dest, crash, out_road = info["arrive_dest"], info["crash"], info["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_road)

        cm = episode.custom_metrics   # alias
        cm["success_rate"]      = float(arrive_dest)
        cm["crash_rate"]        = float(crash)
        cm["out_of_road_rate"]  = float(out_road)
        cm["max_step_rate"]     = float(max_step_rate)

        # agrégats statistiques
        for name in ("velocity", "steering", "acceleration", "step_reward"):
            arr = np.asarray(episode.user_data[name])
            cm[f"{name}_max"]  = float(arr.max())
            cm[f"{name}_mean"] = float(arr.mean())
            cm[f"{name}_min"]  = float(arr.min())

        cm["cost"] = float(sum(episode.user_data["cost"]))

    # ---------- Agrégation après chaque itération ----------
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # (1) Remonter toutes les métriques de sampler_results à la racine
        sampler = result.get("sampler_results", {})
        for k, v in sampler.items():
            # évite d’écraser une clé déjà forcée par l’utilisateur
            result.setdefault(k, v)

        # (2) Longueur moyenne d’épisode : version ≥ 2.4 ou fallback
        length = sampler.get("episode_len_mean", result.get("episode_len_mean"))
        if length is None:
            # calcul de secours à partir de l’histogramme
            hist = sampler.get("hist_stats", {}).get("episode_lengths", [])
            length = float(np.mean(hist)) if hist else np.nan
        result["length"] = length

        # (3) Initialiser les colonnes attendues
        for key in ("success", "crash", "out", "max_step", "cost"):
            result.setdefault(key, np.nan)

        cm = result.get("custom_metrics", {})
        # Ces clés sont créées dans on_episode_end ; si jamais elles manquent,
        # on garde NaN pour ne pas casser les downstream scripts.
        result["success"]   = cm.get("success_rate_mean",   result["success"])
        result["crash"]     = cm.get("crash_rate_mean",     result["crash"])
        result["out"]       = cm.get("out_of_road_rate_mean", result["out"])
        result["max_step"]  = cm.get("max_step_rate_mean",  result["max_step"])
        result["cost"]      = cm.get("cost_mean",           result["cost"])

def train(
    trainer,
    config,
    stop,
    exp_name,
    num_gpus=0,
    test_mode=False,
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    custom_callback=None,
    max_failures=5,
    **kwargs
):
    ray.init(
        num_gpus=num_gpus,
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
    )
    used_config = {
        "callbacks": custom_callback if custom_callback else DrivingCallbacks,  # Must Have!
        "log_level": "DEBUG" if test_mode else "WARN",
    }
    used_config.update(config)
    config = copy.deepcopy(used_config)

    if not isinstance(stop, dict) and stop is not None:
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if keep_checkpoints_num is not None and not test_mode:
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
    progress_reporter = CLIReporter(metric_columns=metric_columns)
    progress_reporter.add_metric_column("success")
    progress_reporter.add_metric_column("crash")
    progress_reporter.add_metric_column("out")
    progress_reporter.add_metric_column("max_step")
    progress_reporter.add_metric_column("length")
    progress_reporter.add_metric_column("cost")
    kwargs["progress_reporter"] = progress_reporter

    if "verbose" not in kwargs:
        kwargs["verbose"] = 1 if not test_mode else 2

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True if "checkpoint_at_end" not in kwargs else kwargs.pop("checkpoint_at_end"),
        stop=stop,
        config=config,
        max_failures=max_failures if not test_mode else 0,
        reuse_actors=False,
        storage_path="/home/o/olbus4/links/scratch/metadrive/metadrive/results",
        **kwargs
    )
    return analysis


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="generalization_experiment")
    parser.add_argument("--num-gpus", type=int, default=0)
    return parser

def env_creator(env_config):
    legacy = createGymWrapper(MetaDriveEnv)(env_config)   # <- env Gym (hérite de gym.Env)
    return GymToGymnasiumWrapper(legacy)  # <- env Gymnasium (hérite de gymnasium.Env)


if __name__ == '__main__':
    args = get_train_parser().parse_args()
    exp_name = args.exp_name
    stop = int(1000_0000)

    register_env("MetaDriveGymnasium-v0", env_creator)

    config = dict(

        # ===== Training Environment =====
        # Train the policies in scenario sets with different number of scenarios.
        env="MetaDriveGymnasium-v0",
        env_config=dict(
            num_scenarios=tune.grid_search([1, 5]),
            start_seed=tune.grid_search([5000]),
            random_traffic=False,
            traffic_density=tune.grid_search([0.1])
        ),

        # ===== Framework =====
        framework="torch",

        # ===== Evaluation =====
        # Evaluate the trained policies in unseen scenarios.
        evaluation_interval=4,
        evaluation_duration=50,
        evaluation_config=dict(env_config=dict(num_scenarios=50, start_seed=0)),
        evaluation_num_env_runners=1,

        # ===== Training =====

        # Parallelism
        num_workers=4,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=0.8,
        num_cpus_for_driver=0.5,
        num_gpus=1 if args.num_gpus != 0 else 0,

        # Hyper-parameters for PPO
        horizon=500,
        rollout_fragment_length=200,
        sgd_minibatch_size=128,
        train_batch_size=4000,
        num_sgd_iter=5,
        lr=3e-4,
        **{"lambda": 0.95},
    )

    train(
        "PPO",
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        test_mode=False
    )
