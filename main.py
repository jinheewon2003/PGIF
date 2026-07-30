"""Entry point: train a PGIF agent on a MuJoCo gym environment, then evaluate it."""
import argparse
import os

import gym

from pgif.agent import Agent
from pgif.play import Play
from pgif.train import Train


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a PGIF agent.")
    parser.add_argument("--env-name", default="Walker2d", help="Gym MuJoCo environment name (without version suffix)")
    parser.add_argument("--n-iterations", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for actor and critic")
    parser.add_argument("--epochs", type=int, default=10, help="Optimization epochs per iteration")
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--horizon", type=int, default=500, help="Rollout length per iteration")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and just evaluate saved weights")
    return parser.parse_args()


def main():
    args = parse_args()
    env_name = args.env_name

    test_env = gym.make(env_name + "-v2")
    n_states = test_env.observation_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
    n_actions = test_env.action_space.shape[0]

    print(f"Number of states: {n_states}\n"
          f"Action bounds: {action_bounds}\n"
          f"Number of actions: {n_actions}")

    if not os.path.exists(env_name):
        os.mkdir(env_name)
        os.mkdir(env_name + "/logs")
    env = gym.make(env_name + "-v2")

    agent = Agent(n_states=n_states, n_iter=args.n_iterations, env_name=env_name,
                  action_bounds=action_bounds, n_actions=n_actions, lr=args.lr)

    if not args.eval_only:
        trainer = Train(env=env, test_env=test_env, env_name=env_name, agent=agent,
                         horizon=args.horizon, n_iterations=args.n_iterations, epochs=args.epochs,
                         mini_batch_size=args.mini_batch_size, epsilon=args.clip_range,
                         initialAlpha=0.0001, initialBeta=0.0001)
        trainer.step()

    player = Play(env, agent, env_name)
    player.evaluate()


if __name__ == "__main__":
    main()
