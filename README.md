# PGIF

An implementation of Policy Gradient in the Future ([arXiv:2108.02096](https://arxiv.org/abs/2108.02096)), a PPO-style actor-critic algorithm for continuous-control MuJoCo environments.

Structure:
```
pgif/
  networks.py   # Actor, Critic, MLP, and BackwardsLSTM
  agent.py       # Agent: networks, optimizers, checkpointing
  train.py       # Rollout collection, GAE, and the training loop
  play.py        # Loads a checkpoint and records a video of the agent
main.py          # CLI entry point
```