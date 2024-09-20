# Required system libraries installation for MuJoCo and other dependencies
!apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common

!apt-get install -y patchelf

# Python libraries installation
!pip install gym
!pip install free-mujoco-py
!pip install mujoco

# Required imports
import mujoco_py
import gym
import numpy as np
from gym import wrappers
import scipy.signal
from datetime import datetime
import os
import argparse
import signal
import torch
from torch import nn, from_numpy, device
from torch.distributions import normal
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from mujoco_py.generated import const
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd

# Utility class for running mean and standard deviation calculations for batch normalization
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        # Update mean and variance using batch data
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # Update mean, variance, and count from batch moments
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    # Combine statistics to update overall mean, variance, and count
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

# Class to handle evaluation and visualization of the trained agent
class Play:
    def __init__(self, env, agent, env_name, max_episode=1):
        self.env = env
        self.max_episode = max_episode
        self.agent = agent
        _, self.state_rms_mean, self.state_rms_var = self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cpu")
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.VideoWriter = cv2.VideoWriter(env_name + ".avi", self.fourcc, 50.0, (250, 250))

    def evaluate(self):
        # Run a set number of episodes to evaluate the agent
        for _ in range(self.max_episode):
            s = self.env.reset()
            episode_reward = 0
            for _ in range(self.env._max_episode_steps):
                # Normalize state input
                s = np.clip((s - self.state_rms_mean) / (self.state_rms_var ** 0.5 + 1e-8), -5.0, 5.0)
                dist, _, _ = self.agent.choose_dist(s)
                action = dist.sample().cpu().numpy()[0]
                s_, r, done, _ = self.env.step(action)
                episode_reward += r
                if done:
                    break
                s = s_

                # Render and save video frames
                I = self.env.render(mode='rgb_array')
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, (250, 250))
                self.VideoWriter.write(I)
            print(f"Episode reward: {episode_reward:.3f}")
        self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()

# Actor network class
class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        # Define the network layers
        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.mu = nn.Linear(in_features=64, out_features=self.n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, inputs):
        # Forward pass for the actor network
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        std = self.log_std.exp()
        dist = normal.Normal(mu, std)
        return dist, mu, std

# Critic network class
class Critic(nn.Module):
    def __init__(self, n_states):
        super(Critic, self).__init__()
        self.n_states = n_states

        # Define the network layers
        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.value = nn.Linear(in_features=64, out_features=1)

        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, inputs):
        # Forward pass for the critic network
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        value = self.value(x)
        return value

# Agent class encapsulating the Actor-Critic model
class Agent:
    def __init__(self, env_name, n_iter, n_states, action_bounds, n_actions, lr):
        self.env_name = env_name
        self.n_iter = n_iter
        self.action_bounds = action_bounds
        self.n_actions = n_actions
        self.n_states = n_states
        self.device = torch.device("cpu")
        self.lr = lr

        # Initialize actor and critic networks
        self.current_policy = Actor(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.critic = Critic(n_states=self.n_states).to(self.device)

        # Optimizers
        self.actor_optimizer = Adam(self.current_policy.parameters(), lr=self.lr, eps=1e-5)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        self.critic_loss = torch.nn.MSELoss()

        # Learning rate schedulers
        self.scheduler = lambda step: max(1.0 - float(step / self.n_iter), 0)
        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=self.scheduler)

    def choose_dist(self, state):
        # Choose action distribution based on the current state
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            dist, mu, std = self.current_policy(state)
        return dist, mu, std

    def get_value(self, state):
        # Get the value prediction from the critic
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            value = self.critic(state)
        return value.detach().cpu().numpy()

    def optimize(self, actor_loss, critic_loss):
        # Optimize the actor and critic networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def schedule_lr(self):
        # Update learning rates
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def save_weights(self, iteration, state_rms):
        # Save model weights
        torch.save({"current_policy_state_dict": self.current_policy.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                    "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                    "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
                    "iteration": iteration,
                    "state_rms_mean": state_rms.mean,
                    "state_rms_var": state_rms.var,
                    "state_rms_count": state_rms.count}, self.env_name + "_weights.pth")

    def load_weights(self):
        # Load saved model weights
        checkpoint = torch.load(self.env_name + "_weights.pth")
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        state_rms_mean = checkpoint["state_rms_mean"]
        state_rms_var = checkpoint["state_rms_var"]

        return iteration, state_rms_mean, state_rms_var

    def set_to_eval_mode(self):
        # Set networks to evaluation mode
        self.current_policy.eval()
        self.critic.eval()

    def set_to_train_mode(self):
        # Set networks to training mode
        self.current_policy.train()
        self.critic.train()

# Main training class handling the training loop
class Train:
    def __init__(self, env, test_env, env_name, n_iterations, agent, epochs, mini_batch_size, epsilon, horizon, initialAlpha, initialBeta):
        self.env = env
        self.env_name = env_name
        self.test_env = test_env
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))
        self.alpha = initialAlpha
        self.beta = initialBeta
        self.increase_inits = 0.000001

        self.rewards_history = []
        self.actor_losses = []
        self.running_reward = 0

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, values, log_probs, meanZs, stdZs, backwards):
        # Generator for mini-batches of training data
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            indices = indices.astype(int)
            yield states[indices], actions[indices], returns[indices], advs[indices], values[indices], log_probs[indices], meanZs[indices], stdZs[indices], [backwards[i] for i in indices]

    def train(self, states, actions, advs, values, log_probs, meanZs, stdZs, backwards):
        # Train the actor and critic networks
        values = np.vstack(values[:-1])
        log_probs = np.vstack(log_probs)
        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        actions = np.vstack(actions)

        # Define a separate MLP model for comparison (optional)
        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        mlp = MLP(n_states, n_actions)
        lf = nn.MSELoss()
        op = torch.optim.Adam(mlp.parameters(), lr=0.001)

        meanZs = np.array(meanZs)
        stdZs = np.array(stdZs)

        for epoch in range(self.epochs):
            # Iterate through mini-batches
            for state, action, return_, adv, old_value, old_log_prob, meanZ, stdZ, backward in self.choose_mini_batch(self.mini_batch_size, states, actions, returns, advs, values, log_probs, meanZs, stdZs, backwards):
                actor_loss = 0
                critic_loss = 0
                lossAxCalc = 0
                for index in range(len(state)):
                    # Convert data to tensors
                    s = torch.Tensor(state[index]).to(self.agent.device)
                    a = torch.Tensor(action[index]).to(self.agent.device)
                    r = torch.Tensor(return_[index]).to(self.agent.device)
                    ad = torch.Tensor(adv[index]).to(self.agent.device)
                    ov = torch.Tensor(old_value[index]).to(self.agent.device)
                    olp = torch.Tensor(old_log_prob[index]).to(self.agent.device)
                    mz = torch.Tensor(meanZ[index]).to(self.agent.device)
                    sz = torch.Tensor(stdZ[index]).to(self.agent.device)
                    bck = backward[index]

                    # Calculate critic loss
                    v = self.agent.critic(s)
                    critic_loss += self.agent.critic_loss(v, r)

                    # Backwards pass and policy updates
                    meanZPG = bck[0]
                    stdZPG = bck[1]
                    distPG = normal.Normal(meanZPG, stdZPG)
                    actionPG = distPG.sample().cpu().numpy()[0]
                    new_log_prob = distPG.log_prob(torch.Tensor(actionPG)).detach()
                    meanAx, stdAx = mlp.forward(s)
                    lossAxCalc += lf(meanAx, meanZPG)
                    lossKL = self.kl_divergence_gaussian(mz, sz, meanZPG, stdZPG)

                    # Calculate actor loss
                    ratio = (new_log_prob - olp).exp()
                    actor_loss += self.compute_actor_loss(ratio, ad, lossAx, lossKL)

                    self.actor_losses.append(actor_loss.item())
                    a_loss = actor_loss.clone()
                    a_loss.requires_grad_()

                # Optimize actor and critic
                self.agent.optimize(a_loss, critic_loss)

        return actor_loss, critic_loss

    @staticmethod
    def kl_divergence_gaussian(mu1, std1, mu2, std2):
        # Calculate the KL divergence between two Gaussian distributions
        kl = 0.5 * (torch.pow(std1 / std2, 2) + torch.pow((mu2 - mu1) / std2, 2) - 1 + 2 * torch.log(std2 / std1))
        return kl.mean()

    def step(self):
        # Main training loop
        state = self.env.reset()
        step_rewards = []
        actor_losses = []

        for iteration in range(1, self.n_iterations + 1):
            meanZs = [] 
            stdZs = [] 
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            backwards = []

            # Generate training data through interactions with the environment
            for t in range(self.horizon):
                state = np.clip((state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                dist, mean, std = self.agent.choose_dist(state)
                action = dist.sample().cpu().numpy()[0]
                log_prob = dist.log_prob(torch.Tensor(action))
                value = self.agent.get_value(state)
                next_state, reward, done, _ = self.env.step(action)

                meanZs.append(mean)
                stdZs.append(std)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state

            next_state = np.clip((next_state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
            next_value = self.agent.get_value(next_state) * (1 - done)
            values.append(next_value)

            # Calculate advantages using Generalized Advantage Estimation (GAE)
            advs = self.get_gae(rewards, values, dones)
            states = np.vstack(states)

            # Backwards pass for training stability
            backwards = self.backwardsPass(states, actions)

            # Train the model and update the policy
            actor_loss, critic_loss = self.train(states, actions, advs, values, log_probs, meanZs, stdZs, backwards)
            self.alpha += self.increase_inits
            self.beta += self.increase_inits
            self.agent.schedule_lr()
            eval_rewards = evaluate_model(self.agent, self.test_env, self.state_rms, self.agent.action_bounds)
            self.state_rms.update(states)
            self.print_logs(iteration, actor_loss, critic_loss, eval_rewards)

            step_rewards.append(eval_rewards)
            actor_losses.append(actor_loss.item())

        # Plot training performance
        df = pd.DataFrame({
            'Step': range(1, self.n_iterations + 1),
            'Reward': step_rewards,
            'Actor Loss': actor_losses
        })

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        df_mean = df.groupby('Step').mean()
        df_std = df.groupby('Step').std()
        plt.plot(df_mean.index, df_mean['Reward'], label='Mean Reward')
        plt.fill_between(df_mean.index, df_mean['Reward'] - df_std['Reward'], df_mean['Reward'] + df_std['Reward'], alpha=0.3)
        plt.xlabel('Step')
        plt.ylabel('Mean Episodic Reward')
        plt.title('Walker2d-v2')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(df['Step'], df['Actor Loss'], label='Actor Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Actor Loss')
        plt.title('Walker2d-v2')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def backwardsPass(self, states, actions):
        # Reverse pass through the states and actions to stabilize training
        backwards = []
        backwardsStates = states[::-1].copy()
        backwardsStates = torch.Tensor(backwardsStates).to(self.agent.device)
        backwardsActions = actions[::-1].copy()
        backwardsActions = torch.Tensor(backwardsActions).to(self.agent.device)

        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]

        lstm = BackwardsLSTM(n_states, n_actions)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

        for index, state in enumerate(backwardsStates):
            mean, log_std = lstm(state)
            backwards.append([mean.detach(), log_std.detach()])

            loss = loss_function(mean, backwardsActions[index].reshape(mean.shape))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach()

        return backwards[::-1]

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        # Calculate advantages using Generalized Advantage Estimation (GAE)
        advs = []
        gae = 0
        dones.append(0)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
        return np.vstack(advs)

    def compute_actor_loss(self, ratio, adv, lossAx, lossKL):
        # Calculate the actor loss
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        loss -= self.alpha * lossAx
        loss -= self.beta * lossKL
        return loss

    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards):
        # Print training logs
        if iteration == 1:
            self.running_reward = eval_rewards
        else:
            self.running_reward = self.running_reward * 0.99 + eval_rewards * 0.01

        if iteration % 25 == 0:
            print(f"Iter: {iteration} | "
                  f"Ep_Reward: {eval_rewards:.3f} | "
                  f"Running_reward: {self.running_reward:.3f} | "
                  f"Actor_Loss: {actor_loss:.3f} | "
                  f"Critic_Loss: {critic_loss:.3f} | "
                  f"Iter_duration: {time.time() - self.start_time:.3f} | "
                  f"lr: {self.agent.actor_scheduler.get_last_lr()}")
            self.agent.save_weights(iteration, self.state_rms)

        # Log metrics to TensorBoard
        with SummaryWriter(self.env_name + "/logs") as writer:
            writer.add_scalar("Episode running reward", self.running_reward, iteration)
            writer.add_scalar("Episode reward", eval_rewards, iteration)
            writer.add_scalar("Actor loss", actor_loss, iteration)
            writer.add_scalar("Critic loss", critic_loss, iteration)

# Function to evaluate the model's performance on the environment
def evaluate_model(agent, env, state_rms, action_bounds):
    total_rewards = 0
    s = env.reset()
    done = False
    while not done:
        s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5.0, 5.0)
        dist, _, _ = agent.choose_dist(s)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        if total_rewards < 0:
            total_rewards = 0
        s = next_state
    return total_rewards

# Main entry point
if __name__ == "__main__":
    # Environment and agent configuration
    ENV_NAME = "Walker2d"
    TRAIN_FLAG = True
    test_env = gym.make(ENV_NAME + "-v2")
    n_states = test_env.observation_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
    n_actions = test_env.action_space.shape[0]
    n_iterations = 500
    lr = 3e-4
    epochs = 10
    clip_range = 0.2
    mini_batch_size = 64
    T = 500

    print(f"Number of states: {n_states}\n"
          f"Action bounds: {action_bounds}\n"
          f"Number of actions: {n_actions}")

    if not os.path.exists(ENV_NAME):
        os.mkdir(ENV_NAME)
        os.mkdir(ENV_NAME + "/logs")
    env = gym.make(ENV_NAME + "-v2")

    # Initialize the agent
    agent = Agent(n_states=n_states, n_iter=n_iterations, env_name=ENV_NAME, action_bounds=action_bounds, n_actions=n_actions, lr=lr)

    if TRAIN_FLAG:
        # Training process
        trainer = Train(env=env, test_env=test_env, env_name=ENV_NAME, agent=agent, horizon=T, n_iterations=n_iterations, epochs=epochs, mini_batch_size=mini_batch_size, epsilon=clip_range, initialAlpha=0.0001, initialBeta=0.0001)
        trainer.step()

    # Evaluate the trained agent
    player = Play(env, agent, ENV_NAME)
    total_reward = player.evaluate()