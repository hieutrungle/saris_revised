import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
import warnings

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["BATCHED_PIPE_TIMEOUT"] = "200"  # to avoid timeout errors in batched pipe
warnings.filterwarnings("ignore", category=UserWarning, module="torchrl")

from torchinfo import summary
from dataclasses import dataclass
import gc
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
from torchrl.envs import ParallelEnv
from tensordict import TensorDict

# Utils
torch.manual_seed(0)
from torchrl.record.loggers import TensorboardLogger, WandbLogger, Logger
from matplotlib import pyplot as plt
from tqdm import tqdm
import functools


class TrainConfig:
    pass


if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


def make_random_ind(bounds: np.ndarray) -> "creator.Individual":
    """
    bounds: shape (2, n_agents, 3) -> [0, :, :]=low, [1, :, :]=high
    returns: numpy.ndarray (n_agents, 3)
    """
    lows = bounds[0, ...]
    highs = bounds[1, ...]
    sample = np.random.uniform(lows, highs)

    # Create individual with proper fitness initialization
    individual = creator.Individual(sample)
    individual.fitness = creator.FitnessMax()
    return individual


class GAReflectorOptimizer:
    """Evolve the best focal-point (x,y,z) for max RSSI."""

    def __init__(
        self,
        config: TrainConfig,
        bounds: Tuple[tuple, tuple],  # tuple(focal_low, focal_high)
        envs: ParallelEnv,
        population: int = 60,
        generations: int = 120,
        seed: int = 2025,
    ):
        # ---- parameters -----------------------------------------------------
        random.seed(seed)
        np.random.seed(seed)
        self.config = config
        self.low_bnd = np.asarray(bounds[0], dtype=np.float32)
        self.high_bnd = np.asarray(bounds[1], dtype=np.float32)
        self.bounds_np = np.stack([self.low_bnd, self.high_bnd])
        self.pop_size = population
        self.gens = generations
        self.envs = envs
        self.n_agents = self.envs.action_spec.shape[-2]
        self.n_envs = self.envs.action_spec.shape[0]

        # ---- DEAP toolbox ---------------------------------------------------
        self._setup_deap()

    # -----------------------------------------------------------------------
    def _setup_deap(self):
        tb = self.toolbox = base.Toolbox()
        tb.register("individual", functools.partial(make_random_ind, self.bounds_np))
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate", self._eval)
        tb.register("mate", self._blend_cx)
        tb.register("mutate", self._gauss_mut)
        tb.register("select", tools.selTournament, tournsize=3, fit_attr="fitness")
        # tb.register("select", tools.selTournament, tournsize=3)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("mean", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

    # -----------------------------------------------------------------------
    def _eval(self, new_focals: torch.Tensor) -> float:
        """Evaluate a batch of individuals.

        Args:
            new_focals (tensor): Individual to evaluate, containing 9 focal point coordinates (num_envs, 1, n_agents, 3).

        Returns:
            float: The average RSSI value for all users in the environment.
        """
        new_focals = TensorDict(
            {"agents": {"action": new_focals}}, batch_size=(*new_focals.shape[:2],)
        )
        new_focals = new_focals.to(self.envs.device)
        td = self.envs.step(new_focals)
        cur_rss = td.get(("next", "agents", "cur_rss"))
        cur_rss = 10 * torch.log10(cur_rss) + 30

        cur_rss = [cur_rss[:, 0, i, i] for i in range(cur_rss.shape[-1])]
        cur_rss = torch.stack(cur_rss, dim=-1)
        tot = torch.mean(cur_rss, dim=-1)
        tot = list(tot.detach().cpu().numpy())
        return tot, cur_rss.detach().cpu().numpy()

    # ------------- GA primitives -----------------
    def _blend_cx(self, c1, c2, alpha=0.2):
        # print(f"\nInitial c1: {c1}, c2: {c2}")
        c1 = copy.deepcopy(c1)
        c2 = copy.deepcopy(c2)
        gamma = np.random.uniform(-alpha, 1 + alpha, size=c1.shape)
        inv_g = 1.0 - gamma
        child1 = inv_g * c1 + gamma * c2
        child2 = gamma * c1 + inv_g * c2
        # print(f"Before child1: {child1}, child2: {child2}")
        np.clip(child1, self.low_bnd, self.high_bnd, out=child1)
        np.clip(child2, self.low_bnd, self.high_bnd, out=child2)
        # print(f"After child1: {child1}, child2: {child2}")
        c1[:] = child1
        c2[:] = child2
        # print(f"Final c1: {c1}, c2: {c2}\n")
        return c1, c2

    def _gauss_mut(self, ind, sigma_ratio=0.1, indpb=0.20):
        # print(f"\nBefore mutation: {ind}")
        mutant = copy.deepcopy(ind)
        mask = np.random.rand(*mutant.shape) < indpb
        widths = (self.high_bnd - self.low_bnd) * sigma_ratio
        mutant[mask] += np.random.normal(0.0, widths[mask])
        np.clip(mutant, self.low_bnd, self.high_bnd, out=mutant)
        # print(f"After mutation: {mutant}")
        return (mutant,)

    # -----------------------------------------------------------------------
    def train(self, train_idx: int = 0) -> Tuple["creator.Individual", float]:

        self.pre_train_config()
        pop, fits, rss, td = self.init_population()
        # for ind in pop:
        #     print(f"Individual: {ind}, Fitness: {ind.fitness.values}")

        # evolutionary loop --------------------------------------------------
        cxpb, mutpb = 0.7, 0.2
        for g in range(self.gens):
            # selection → variation
            offspring = list(map(self.toolbox.clone, self.toolbox.select(pop, len(pop))))
            offspring = [copy.deepcopy(o) for o in offspring]
            self.perform_crossover_and_mutation(cxpb, mutpb, offspring)

            # re-evaluate new individuals
            self.evaluate_invalid_individuals(rss, offspring)

            # pop = list(map(self.toolbox.clone, self.toolbox.select(offspring, len(pop))))
            pop[:] = offspring
            self.hof.update(pop)
            best = self.hof[0]

            # Record statistics
            rec = self.stats.compile(pop)
            for k, v in rec.items():
                self.hist[k].append(v)

            # for ind, r in zip(pop, rss):
            #     print(f"Individual: {ind}\nFitness: {ind.fitness.values}\nRSSI: {r}\n")
            # print()
            # print(f"generation {g}: {rec}")

            if g % 1 == 0:
                print(
                    f"G{g:03d}  avg={np.round(rec['mean'], 2)};  max={np.around(rec['max'], 2)};  best={np.around(best.fitness.values, 2)}"
                )

            # Check for improvement and stop if no improvement
            current_best = -np.inf
            for ind, r in zip(pop, rss):
                if np.mean(ind.fitness.values) > np.mean(current_best):
                    current_best = ind.fitness.values
                    current_best_rss = r
            if not self.check_improvement(current_best, current_best_rss):
                break
            print()

        print(f"Best focal-point: {best}")
        print(f"Best RSSI values: {self.best_rss}")
        print(f"RSSI values: {best.fitness.values}")
        # save the best individual, rssi, td, and history
        if self.config.checkpoint_dir != "-1":
            save_data = {
                "best_focal_point": best,
                # "best_rssi": self.best_rss,
                "td": td,
                "history": self.hist,
            }
            save_path = os.path.join(self.config.checkpoint_dir, f"best_focal_point_{train_idx}.pt")
            torch.save(save_data, save_path)
        return best, best.fitness.values[0]

    def evaluate_invalid_individuals(self, rss, offspring):
        invalids = []
        idxs = []
        for idx, ind in enumerate(offspring):
            if not ind.fitness.valid:
                invalids.append(ind)
                idxs.append(idx)

        # invalids = [i for i in offspring if not i.fitness.valid]
        # pad invalids to match multiple of num_envs
        n_invalid = len(invalids)
        if n_invalid % self.n_envs != 0:
            n_pad = self.n_envs - (n_invalid % self.n_envs)
            invalids += [self.toolbox.clone(invalids[0]) for _ in range(n_pad)]
            idxs += [idxs[0] for _ in range(n_pad)]

        invalid_tensor = torch.tensor(list(invalids), dtype=torch.float32, device=self.envs.device)
        invalid_tensor = invalid_tensor.reshape(invalid_tensor.shape[0], 1, self.n_agents, 3)
        invalid_tensor = invalid_tensor.reshape(-1, self.n_envs, *invalid_tensor.shape[1:])

        invalid_fits_rss = list(map(self.toolbox.evaluate, invalid_tensor))
        invalid_fits = []
        invalid_rss = []
        for f, r in invalid_fits_rss:
            invalid_fits.extend(f)
            invalid_rss.extend(r)
        invalid_fits = invalid_fits[:n_invalid]
        invalid_rss = invalid_rss[:n_invalid]

        # # evaluate invalids individuals
        for idx, ind, f, r in zip(idxs, invalids, invalid_fits, invalid_rss):
            ind.fitness.values = (f,)
            rss[idx] = r

    def perform_crossover_and_mutation(self, cxpb, mutpb, offspring):

        # for o in offspring:
        #     print(f"Offspring: {o}, Fitness: {o.fitness.values}")
        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                self.toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # mutation
        for m in offspring:
            if random.random() < mutpb:
                self.toolbox.mutate(m)
                del m.fitness.values

        # for o in offspring:
        #     print(f"Offspring: {o}, Fitness: {o.fitness.values}")

    def init_population(self) -> List[creator.Individual]:
        pop = self.toolbox.population(n=self.pop_size)
        pop_tensor = torch.tensor(
            np.array(copy.deepcopy(pop)), dtype=torch.float32, device=self.envs.device
        )
        # reshape population to match the expected input shape
        pop_tensor = pop_tensor.reshape(-1, self.n_envs, 1, *pop_tensor.shape[1:])
        td = self.envs.reset()
        fits_rss = list(map(self.toolbox.evaluate, pop_tensor))
        fits = []
        rss = []
        for f, r in fits_rss:
            fits.extend(f)
            rss.extend(r)

        # fits = list(np.array(fits).flatten())
        for ind, f in zip(pop, fits):
            # print(f"Individual: {len(ind.fitness.weights)}, Fitness: {f}")
            ind.fitness.values = (f,)
            # print(f"Individual: {ind}, Fitness: {ind.fitness.values}")
        #     print(f"Individual: {ind}, Fitness: {ind.fitness.values}")
        # print()
        return pop, fits, rss, td

    def pre_train_config(self):

        # ---- tracking -------------------------------------------------------
        self.hist = defaultdict(list)
        self.hof = tools.HallOfFame(1, similar=np.array_equal)

        # monitor the improve in the best fitness value
        self.best_fitness = -np.inf
        self.best_rss = None
        self.no_improvement = 0
        self.max_no_improvement = 5  # number of generations without improvement before stopping

    def check_improvement(self, current_fitness: np.ndarray, rss: List[float]) -> bool:
        """Check if the current fitness is better than the best fitness."""
        if np.mean(current_fitness) > np.mean(self.best_fitness):
            self.best_fitness = current_fitness
            self.best_rss = rss
            self.no_improvement = 0
        else:
            self.no_improvement += 1

        if self.no_improvement >= self.max_no_improvement:
            print(f"No improvement for {self.max_no_improvement} generations, stopping training.")
            return False
        return True

    # -----------------------------------------------------------------------
    def plot(self):
        if not self.hist["mean"]:
            print("Run train() first.")
            return
        gens = range(len(self.hist["mean"]))
        plt.figure(figsize=(10, 4))
        plt.plot(gens, self.hist["mean"], label="Avg")
        plt.plot(gens, self.hist["max"], label="Best")
        plt.fill_between(
            gens,
            np.array(self.hist["mean"]) - np.array(self.hist["std"]),
            np.array(self.hist["mean"]) + np.array(self.hist["std"]),
            alpha=0.3,
        )
        plt.xlabel("Generation")
        plt.ylabel("RSSI (dBm)")
        plt.title("GA Convergence")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


def run_ga(envs: ParallelEnv, config: "TrainConfig"):

    print(f"=" * 30 + "Genetic Algorithm" + "=" * 30)

    focal_lows = envs.observation_spec["agents", "focals"].low[0, 0].detach().cpu()
    # focal_lows = torch.ones_like(focal_lows, device=config.device) * (-6.5)
    focal_lows = focal_lows.cpu().numpy()
    focal_lows[..., 2] = -1.0  # z-coordinate should be 0.0
    focal_highs = envs.observation_spec["agents", "focals"].high[0, 0].detach().cpu()
    # focal_highs = torch.ones_like(focal_highs, device=config.device) * 6.5
    focal_highs = focal_highs.cpu().numpy()
    focal_highs[..., 2] = 3.0  # z-coordinate should be 0.0
    bounds = tuple([focal_lows, focal_highs])  # bounds for x, y, z

    try:
        if envs.is_closed:
            envs.start()

        # Train the GA
        ga_reflector_optimizer = GAReflectorOptimizer(
            config=config,
            bounds=bounds,  # Example bounds for x, y, z
            envs=envs,
            population=config.num_envs * 20,
            generations=20,
            seed=config.seed,
        )
        epochs = 21
        pbar = tqdm(total=epochs)
        for train_idx in range(epochs):

            # Train the GA
            start_time = time.perf_counter()
            best_focal_point, best_fitness = ga_reflector_optimizer.train(train_idx)
            end_time = time.perf_counter()
            print(f"Training time: {end_time - start_time:.2f} seconds\n")
            # pbar.set_description(f"best_fitness = {best_fitness}", refresh=False)
            pbar.update()

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
